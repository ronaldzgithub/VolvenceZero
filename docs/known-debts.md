# Known Architecture Debt

> Status: tracked, not blocking
> Last updated: 2026-05-10 (Real-Person Figure Vertical F1-F6 全套 land；新增 #18-#23 为 figure-vertical 训练数据 / 训练管线 / DLaaS adopt 接线层未来工作)

> 2026-05-10 update (Real-Person Figure Vertical F1-F6): 16 packets land；新 wheel `lifeform-domain-figure` + `vz-substrate` 一处 additive `persona_lora_pool.py` + `lifeform-expression` 三个 enforcer (`GroundedDecoder` / `ScopeRefuser` / `StylePriorInjector`) + `dlaas-platform-{contracts,registry}` 4 个 figure 字段 + `lifeform-service` `figure_bundle_store.py` 全套到位；L1 (style prior) / L3 (citation grounding) / L4 (scope refusal) 在零 GPU 训练下端到端可演示，L2 (steering) + L1+L2 (persona LoRA) 通过 OFFLINE `ModificationGate` 走 SHADOW。1063/1063 contract / smoke / e2e test 全绿，含 `test_full_chain_e2e_smoke.py` 一次性把 retrieval × coverage × style × steering × LoRA 在 Einstein bundle 上跑通；详见 [`docs/specs/figure-vertical.md`](specs/figure-vertical.md) + [`docs/DATA_CONTRACT.md`](DATA_CONTRACT.md) §2.15。**遗留三块需要后续 follow-up（已开 debt）**：(a) F5 steering 用 CPU contrastive linear readout 在 hashing-embedding 坐标系（→ #21）+ F6 LoRA 用 deterministic synthetic backend（→ #18）；(b) 基础数据准备只到 V1 archive schema，没有 V2 HTTPArchiveFetcher、没有 PDF/HTML/wiki/OCR parser、没有 curated payload 数据集（→ #19）；(c) 训练管线 script 完全没有，所有 bake / gate apply / rollback 只是 Python 函数没 CLI（→ #23）。DLaaS adopt 主路径未自动 hook-in（→ #22）；PersonaLoRAPool 真热插未接（→ #20）。

> 2026-05-10 update (DLaaS Slice 1 → 7): 6 个新 wheel `dlaas-platform-{contracts,registry,launcher,ops,eval,api}` + 一处 `lifeform-service` 路由扩展全部就位，控制面 + 多渠道 typed envelope + ops + eval gate 完整可演示。Slice 7 测试集中收口落地：`tests/contracts/test_dlaas_dispatch_contracts.py` (22) + `tests/service/test_dlaas_chat_smoke.py` (4) + `tests/service/test_dlaas_multi_tenant_persistence.py` (6) + `tests/service/test_dlaas_full_lifecycle.py` (3) + `tests/service/test_dlaas_backward_compat.py` (8) + `tests/contracts/test_import_boundaries.py` (865) = **908/908 全绿**；`git diff --stat HEAD -- packages/vz-* packages/lifeform-*` 输出**空**——R2/R4/R8 三条铁律的"vz-* / lifeform-* 不动"承诺在 git 层面可验证。Slice 5.4 真流式 SSE 在 rollout 阶段 cancel 以保护 `vz-substrate` 不被改；连同其他 5 条 DLaaS 平台层后续 evolution 沿 follow-up 节奏列入下方 #12-#17。详见 [`docs/specs/dlaas-platform.md`](specs/dlaas-platform.md) + [`docs/moving forward/dlaas-platform-rollout.md`](moving%20forward/dlaas-platform-rollout.md)。

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

## ~~9. `agent/session.py` 与 `application/runtime.py` 仍是 god 文件（W5 残留）~~ —— 2026-05-09 关闭（wave-1 + wave-2 全部 land）

- **路径**：
  - ~~`packages/vz-runtime/src/volvence_zero/agent/session.py`（W5 后约 3825 行）~~ → wave-1 split landed 2026-05-09 (mixin 拆分)，详见下方 update
  - ~~`packages/vz-application/src/volvence_zero/application/runtime.py`（约 3941 行）~~ → wave-2 split landed 2026-05-09 (category 拆分 + re-export shell)，详见下方 update
- **问题**（已解决）：W5 of ssot-cleanup-p0-p4 抽出 `session_helpers.py` (260 行) 与 `application/scoring_helpers.py` (139 行) 的纯函数，但 `AgentSessionRunner` / `ResponseAssemblyModule` 等核心 class 仍住在单文件里，单文件 ≥ 3800 行。
- **违反**（已解决）：可演化性 / 可读性，非 R8 硬违反。
- **短期风险**：~~低~~ → 已解决。
- **触发条件**：~~再有一个大型 feature 落到 `AgentSessionRunner` / `ResponseAssemblyModule` 上时 → 单文件超过 4500 行~~ → 已通过两 wave 拆分提前消除。
- **推荐修法**：~~mixin / 服务对象重组~~ → 已实施（wave-1 mixin、wave-2 category re-export shell）。
- **优先级**：~~低，等 W6+ wave 一起做~~ → 已收尾。

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
> ~~**未做（wave-2 deferred）**：`application/runtime.py` (~3941 行) 是不同 shape——god FILE not god CLASS~~ → 2026-05-09 同日完成，详见下面 wave-2 update。
>
> **2026-05-09 update (wave-2 / `application/runtime.py` category split)**：按 wave-2 plan 实施 category 拆分。这次的 shape 与 wave-1 不同——god FILE not god CLASS——所以用**re-export shell** 模式而不是 mixin 模式：原 `runtime.py` 替换为 ~34 行的 thin re-export shell（`from volvence_zero.application.modules import *`、`from volvence_zero.application.types import *` 等），保持 `from volvence_zero.application.runtime import X` 导入路径对所有 consumers（kernel、tests、lifeform-* packages、examples）零变更。
>
> **行数对比（wave-2）**：
> - `application/runtime.py`：3941 → **34** 行（-3907，god 文件削减 ~99%；剩下的 34 行就是 re-export shell + module docstring）
> - 新增 [`application/types.py`](../packages/vz-application/src/volvence_zero/application/types.py)：682 行（50 dataclasses + 9 enums，原 lines 55-676）
> - 新增 [`application/runtime_helpers.py`](../packages/vz-application/src/volvence_zero/application/runtime_helpers.py)：1686 行（80+ helper functions + 12 module-level prototype constants，原 lines 678-2237。`__all__` 显式导出含 leading-underscore 的 private helpers 以让 modules/* 通过 `from runtime_helpers import *` 拿到 `_application_brief` / `_continuum_*` / `_case_*` / `_response_*` 等）
> - 新增 [`application/rare_heavy_state.py`](../packages/vz-application/src/volvence_zero/application/rare_heavy_state.py)：187 行（`ApplicationRareHeavyState` class，原 lines 2240-2366）
> - 新增 [`application/modules/`](../packages/vz-application/src/volvence_zero/application/modules/) 子包（27 行 `__init__.py` re-export + 8 个 sibling 文件，每个一个 owner Module）：
>   - `experience_fast_prior.py` (265 行) / `retrieval_policy.py` (294 行) / `domain_knowledge.py` (196 行) / `case_memory.py` (330 行) / `strategy_playbook.py` (222 行) / `boundary_policy.py` (232 行) / `response_assembly.py` (339 行) / `experience_consolidation.py` (133 行)
>
> **副带闭合的 latent 问题**：原 `runtime.py` 有 4 处 inline `from volvence_zero.application.scoring_helpers import (...)` 散在 dataclass 段与 helper 段之间。slicer 把第一个 (dedupe) 块切到了 `types.py` 而 helper 段需要它——wave-2 中显式补回 `from volvence_zero.application.scoring_helpers import dedupe as _dedupe` 到 `runtime_helpers.py`，闭合一个会在 `_entry_risk_markers` / `_case_hit_ordering` 等少见 path 触发的 NameError 隐患。
>
> **Test 调整**：[`tests/contracts/test_application_no_regime_id_branching.py`](../tests/contracts/test_application_no_regime_id_branching.py) 两处更新：(a) `_ALLOWED_HARDCODED_HITS` 唯一一条 entry 从 `runtime.py` 改为 `runtime_helpers.py`（hit 跟着 helpers 一起搬家了）；(b) 测试 parametrize 从 `glob("*.py")` 改为 `rglob("*.py")` 并用 relative path，让新加的 `modules/*.py` 也被静态 SSOT 扫描覆盖（原本只扫顶层文件）。
>
> **回归证据**：1204/1204 `tests/contracts/`（包括新加进 rglob 扫描的 ~33 个新 module 文件 parametrize 用例）+ 67/67 `tests/test_application_storage.py` + `tests/test_domain_experience.py` + 4 个 focused `tests/lifeform_e2e/` 测试文件（multi-turn / companion-regime / family-report / LLM evidence chain，覆盖 application 模块在主路径上的全部使用点）。
>
> **未来工作**：runtime_helpers.py 仍是 1686 行，未来可以按 category（continuum / regime / knowledge / case / response）进一步细分；但当前没有触发条件（不是 god FILE），延迟到有需要时再做。

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

## 12. DLaaS Slice 5.4 真流式 SSE 未启用（substrate streaming additive 接口）

- **路径**：
  - 当前实现（一次性 JSON）：[`packages/dlaas-platform-api/src/dlaas_platform_api/dispatch.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/dispatch.py)（chat / teach / task 三个 handler 直接 `await session.run_turn(...)` 后整段返回）
  - 当前 SSE 仅在 admin ledger：[`packages/dlaas-platform-ops/src/dlaas_platform_ops/routes.py`](../packages/dlaas-platform-ops/src/dlaas_platform_ops/routes.py)（`_handle_conversations_stream`，是 ops 事件广播，不是 token-level 输出流）
  - 缺位的 substrate hook：[`packages/vz-substrate/src/volvence_zero/substrate/`](../packages/vz-substrate/src/volvence_zero/substrate/)（`OpenWeightResidualRuntime.generate(...)` 是 sync block；没有 `generate_async` / `stream_tokens` 接口）
- **问题**：DLaaS 公开 API 文档（`DLAAS_README.md` §"Send A Chat Interaction"）说返回可以是 SSE `event: ack/act/chunk/done`；当前实现只在 client `output_contract.stream=true` 时仍返回整段 JSON 一次性回复，未拆 token chunks。Slice 5.4 在 rollout 阶段 cancel 以保护 vz-* 不被动；该项是 DLaaS 6 切片中**唯一可能动 `vz-substrate` 的位置**。
- **违反**：纯产品 UX 体验差，不违反 R2/R4/R8 任何铁律——cancel 的理由是"动 substrate 需要单独 review"，不是动了会出错。
- **风险**：低-中。短期看 mobile / web shell 用户感受到的是"chat 必须等完整生成完才显示"，体验上比真流式差；某些 long-form 输出（report 生成 / 长解释）会让用户以为系统卡住。**不影响功能正确性**。
- **触发条件**：(a) 第一个真实生产集成提出"必须 token-level 流式"的需求；(b) 某个 vertical 的 chat 平均生成时间稳定 > 5s；(c) 接 LLM judge 后发现 evaluation 端的 token 流也需要流式 readout（关联 #13）
- **推荐修法**：
  1. `vz-substrate` 加 additive `async def generate_async(self, prompt, *, on_chunk: Callable[[str], None]) -> str` 接口（不删 / 不改现有 `generate(...)`，新方法独立测试套）
  2. `lifeform-expression.LifeformLLMResponseSynthesizer` 加 `synthesize_streaming(...)` 派生方法
  3. `dlaas-platform-api/dispatch.py` 的 chat / teach / task handler 检测 `envelope.output_contract.stream=True` 时改走 SSE writer：`event: ack` → 多个 `event: chunk` → `event: act`（最终结构化结果）→ `event: done`
  4. 单独 packet review，按 `cursor-convergence-workflow.mdc` 走 SHADOW → ACTIVE
- **优先级**：低（产品 UX 优化；核心架构不依赖）

## 13. DLaaS eval gate 用 fail-closed `DefaultRubricGrader` 占位，未接真实 LLM judge

- **路径**：
  - 占位实现：[`packages/dlaas-platform-eval/src/dlaas_platform_eval/grader.py`](../packages/dlaas-platform-eval/src/dlaas_platform_eval/grader.py)（`DefaultRubricGrader.grade(...)` 给每个 criterion 打 `max_score * 0.5`；`RubricGrader` Protocol 是插件位）
  - 消费者：[`packages/dlaas-platform-eval/src/dlaas_platform_eval/routes.py`](../packages/dlaas-platform-eval/src/dlaas_platform_eval/routes.py)（`_finalize_run` 调 `bundle.grader.grade(...)` 计算 weighted_score）
  - 公共契约：[`packages/dlaas-platform-contracts/src/dlaas_platform_contracts/eval.py`](../packages/dlaas-platform-contracts/src/dlaas_platform_contracts/eval.py)（`RubricEntry` / `ExamSubmissionScore.rubric_breakdown`）
- **问题**：当前 grader 给所有非空响应打 50% × max_score。这意味着：
  - 自动 `POST /dlaas/exam_runs/{id}/execute` 永远过不了 default `pass_threshold=0.6`，license 自动 `granted=False`
  - 只有 `POST /dlaas/exam_runs/{id}/complete` 用 caller-supplied `ai_responses` + 操作员 / 真实 LLM judge 评分时才能 grant license
  - 若未来生产 traffic 默认走 `execute` 路径而不是 `complete`，整个 license gate 形同虚设——整轮自动 exam 全部 0.5 evenly，gate threshold 调到 0.4 也只是把 "全 pass" 的伪装移到不同水位
- **违反**：R12（evaluation 是 readout，不能是学习源）和 OA-1（LLM judge 不能反向写 reward）的精神保留——`DefaultRubricGrader` 是 readout，没有反向写回任何 owner。但其 readout 的**信息含量**（每个 criterion 都打 0.5）使得 license gate 的判别能力为零。
- **风险**：中。架构正确（fail-closed），但**"license = 真实通过 exam evidence"承诺在 grader 接入前不成立**；任何把当前 license 当作"产品就绪"的 cite 都是误导。
- **触发条件**：(a) 第一个 tenant 想跑 launch_gate 之前需要真自动评分；(b) 某个产品 SLO 与 license granted 比例挂钩（无意义指标因为现在永远 not-granted）；(c) 接 LLM judge 后想 cite "DLaaS 已具备 R12 readout-only eval"——必须先把 grader 实例化为真 LLM
- **推荐修法**：
  1. 在 `dlaas-platform-eval` 加 `LLMRubricGrader(provider, prompt_template, parse_strategy)` 实现 `RubricGrader` Protocol，输入 rubric + ai_response + reference_answer，调 LLM 产 per-criterion `score` + 自由文本 `rationale`，解析失败 fail-loud（不走 default 0.5）
  2. `attach_eval_routes(app, *, registry, grader=...)` 在 `build_dlaas_app(...)` 入口暴露 `eval_grader: RubricGrader | None = None` 参数，默认仍是 `DefaultRubricGrader` 但生产部署传 `LLMRubricGrader`
  3. 加诊断 metric：`exam_runs.grader_provider_label` （"default" / "llm:qwen-1.5b" / 等）字段进 `ExamRunSpec.submissions[].rubric_breakdown[i]['grader_label']`，让 license-evaluate 端点能区分"无 evidence 因为没 grader"vs"有 evidence 但 not granted"
  4. 守 OA-1：grader 输出**不**反向写任何 kernel owner（`PERequest` / `RewardingState` / `Face` 等），加 `tests/contracts/test_dlaas_eval_no_kernel_writeback.py` 静态守门
- **优先级**：中（首次想给 license gate "真实可信"语义时强行触发）

## 14. DLaaS audience analysis 是占位 readout，未真正分析 corpus

- **路径**：
  - 占位实现：[`packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py)（`_handle_audience_analyze` 持久化 `cohort_name` + `asset_ids` + 调用方传入的 `communication_style` / `emotion_triggers` / `decision_patterns`，**没真的从 asset 内容提取**任何字段，`evidence_stats` 仅记 `asset_count` + `default_grader=True`）
  - 持久层：[`packages/dlaas-platform-registry/src/dlaas_platform_registry/eval_store.py`](../packages/dlaas-platform-registry/src/dlaas_platform_registry/eval_store.py)（`EvalStore.upsert_audience_profile`）
  - 公共契约：[`packages/dlaas-platform-contracts/src/dlaas_platform_contracts/eval.py`](../packages/dlaas-platform-contracts/src/dlaas_platform_contracts/eval.py)（`AudienceProfileSpec`）
- **问题**：DLaaS 公开 README §"Audience Analysis" 承诺 audience profile 包含从 asset corpus 提取的 `common_questions` / `communication_style` / `emotion_triggers` / `decision_patterns` + `evidence_stats`。当前实现：
  - 不读 asset.uri 内容
  - 不调任何 LLM / NLP 分析
  - 只把 caller-supplied 字段原样存回（等于 caller 自己声明 cohort，不是平台分析）
  - readiness gate 不依赖 audience profile，所以这个端点目前只是"声明 + 持久化"，对其他流程零影响
- **违反**：R8 unchanged（profile 由 platform-registry 单独 owner），但**功能上**这个端点的语义没有 backed by evidence——它声明自己是 audience analysis 但其实只是表单。
- **风险**：低。短期看不影响 lifecycle 任何下游环节；长期看任何 cite 这个端点为"DLaaS 已具备 audience 分析"的文档都是误导。
- **触发条件**：(a) 第一个 vertical 想用 audience profile 的字段驱动 template patch / persona 调整时；(b) 接入 #13 LLM judge 后想统一 audience pipeline 与 grader 共用 LLM provider；(c) 跑产品 demo 时 stakeholder 问"这个 cohort 怎么算出来的"
- **推荐修法**：
  1. 在 `dlaas-platform-eval` 加 `AudienceAnalyzer(provider)` 协议；实现 default `NoOpAudienceAnalyzer`（return empty + `analyzer_label="noop"`）和 `LLMAudienceAnalyzer`（用 prompt 让 LLM 从 asset 内容抽 topics / styles / emotions / decision patterns）
  2. `_handle_audience_analyze` 改为：(a) 解析 `asset_ids` → 从 `AssetStore.get(...)` 拿 asset.uri；(b) 用 `lifeform-ingestion.envelope_from_text(...)` 拉文本；(c) 调 `analyzer.analyze(corpus_chunks)` 拿结构化 profile；(d) 持久化时显式标 `evidence_stats['analyzer']` 字段
  3. 守 R12：analyzer 输出**只**写 audience_profiles 表，**不**反向写 kernel；不接入任何 reward / Face 路径
- **优先级**：低（独立产品功能；不阻塞 lifecycle 主路径）

## 15. DLaaS Activate 用 persona/seed 文本作为 ingestion，未真正抓 asset.uri

- **路径**：
  - 占位实现：[`packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py)（`_activation_text(template, seed_override)` 拼 persona_spec + seed_config 字段，**不读** linked asset.uri 内容）
  - linked assets 的实际位置：[`packages/dlaas-platform-registry/src/dlaas_platform_registry/assets.py`](../packages/dlaas-platform-registry/src/dlaas_platform_registry/assets.py)（`AssetStore.list_template_links` 返回 `TemplateAssetLinkSpec`，asset.uri 在 `AssetStore.get(asset_id).uri`，但 activate 路径没用）
  - 期望路径：sliding to multi-source `IngestionPipeline.process_envelope(envelope_from_text|pdf|docx|...)`
- **问题**：当前 activate 只把 template 自己的 persona + seed 拼一段几百字的 corpus 喂给 IngestionPipeline。这意味着：
  - readiness counters（`world_nodes` / `self_nodes` / `l2_cards`）反映的是 persona text + seed config 大小，而**不**是 tenant 上传的真实训练材料量
  - tenant 把 100MB 训练 chatlog 链给 template，readiness 不会因此变化（除非内容真被 ingest）
  - readiness gate 通过的 template，未必"真的吸收了" tenant 提供的 asset corpus
- **违反**：DLaaS README 约定（asset 上传 + 链到 template + activate 应触发 ingestion）现在没兑现 asset 部分。R8 / R2 / R4 不违反（都是平台层内部行为）。
- **风险**：低-中。lifecycle 走得通（test_full_lifecycle 全绿），但 readiness 信号失真——template 即使没 link 任何 asset 也能 activate 通过。任何 cite "readiness counter ≥ N 说明 corpus 已吸收"的 SLA 都不成立。
- **触发条件**：(a) 第一个生产 tenant 上传 ≥ 10MB 的真实训练材料；(b) 用 readiness counter 作为"训练量计费"维度；(c) 接 #14 audience pipeline 后两套都需要 fetch asset.uri，应该统一抽 `AssetFetcher` 复用
- **推荐修法**：
  1. 在 `dlaas-platform-registry` 或新 `dlaas-platform-asset-fetcher` 模块加 `AssetFetcher` 协议；实现：
     - `LocalFileAssetFetcher` (uri.startswith("file://"))
     - `S3AssetFetcher` (uri.startswith("s3://")，可选依赖 boto3)
     - `HttpAssetFetcher` (uri.startswith("http"))
     - `InlineFetcher`（uri 是 `dlaas:` 开头的占位，从 source_meta 拿 inline_text，用于测试）
  2. `_handle_activate_template` 改为：(a) `AssetStore.list_template_links(template_id)` → asset.id 列表；(b) `AssetStore.get(asset_id)` → uri 列表；(c) `AssetFetcher.fetch_text(uri)` → text；(d) 用 `envelope_from_text` 或 `envelope_from_pdf_*` 按 mime_type 派生 ingestion envelope；(e) 把 persona/seed text 作为 fallback / 补充 chunk 而不是 sole content
  3. `activation_stats` 加 `assets_processed` / `bytes_ingested` / `chunks_total` 字段，让 readiness 真的反映 corpus 吸收量
  4. 单独 packet；与 #14 audience analysis 复用 `AssetFetcher`
- **优先级**：中（生产化阻塞项，但 demo / CI 可用 inline）

## 16. DLaaS contract.tool_policy_snapshot 未推到 AffordanceRegistry 运行时白名单

- **路径**：
  - 持久层：[`packages/dlaas-platform-registry/src/dlaas_platform_registry/contracts.py`](../packages/dlaas-platform-registry/src/dlaas_platform_registry/contracts.py)（`ContractStore.set_ai_id(tool_policy_snapshot=...)` 写入 contracts 表）
  - 计算 snapshot：[`packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py)（`_compute_tool_policy_snapshot(engine_tools)` 派生 `enabled_capabilities` 列表）
  - **缺位的消费者**：`lifeform-affordance.AffordanceRegistry` 应该在 dispatch 时查询 ai_id → contract → tool_policy_snapshot.enabled_capabilities，但当前 invoker 不读
  - launcher 持有 SessionManager 但未注入 per-ai_id capability 白名单：[`packages/dlaas-platform-launcher/src/dlaas_platform_launcher/instance_manager.py`](../packages/dlaas-platform-launcher/src/dlaas_platform_launcher/instance_manager.py)
- **问题**：当前 `POST /dlaas/adopt` 把 `engine_tools={"web_search": True, "data_query": {...}, ...}` 持久化到 contract.tool_policy_snapshot；但运行时 dispatch 一条 chat envelope 时，kernel 通过 `lifeform-affordance` 调工具的路径**不查 contract**。结果：
  - 任何 vertical 启用了某 tool（如 `web_browse`）就会被所有 ai_id 共用，contract 里 `web_browse=False` 不起作用
  - DLaaS README §"engine_tools / tool_policy_snapshot" 承诺的"per-tenant per-contract 工具白名单"现在只是声明性的，运行时不强制
- **违反**：R8 不违反（platform 层是 SSOT），但**功能上**安全护栏未生效——能力降级 / safety 路径全失效
- **风险**：中。如果 tenant A 的 contract 禁了 `web_browse` 但 vertical 默认开了，tenant A 实例仍能通过 affordance 触发外部访问——这是**信任边界违反**。短期 demo 看不出，但生产化前必修。
- **触发条件**：(a) 第一个 tenant 要求"禁用某能力"且需要审计；(b) 同一进程里两个 tenant 的 contract tool policy 不同；(c) 出现 tool-call 引发的安全事件需要溯源
- **推荐修法**：
  1. `dlaas-platform-launcher.InstanceManager` 在 `acquire(ai_id, runtime_template_id, ...)` 时多收一个 `tool_policy_snapshot` 参数，构造 `SessionManager` 时 wrap `AffordanceRegistry` 加 capability 白名单 filter
  2. `lifeform-affordance` 加 `AffordanceRegistry.with_allowlist(enabled_capabilities: tuple[str, ...])` 派生方法（additive，不改原 registry 行为）
  3. `_handle_adopt` 在 `instance_manager.acquire(...)` 调用处把 `final_contract.tool_policy_snapshot["enabled_capabilities"]` 传下去
  4. 运行时 `lifeform-affordance.invoker` 调用前检查 `capability not in allowlist → degrade to text + degraded=True + original_capability=cap`（现有 OutputAct degradation 路径已支持）
  5. 合规审计层加 `tests/service/test_dlaas_tool_policy_enforcement.py`：tenant A 禁 `web_browse`，dispatch chat 时若 vertical 试图调 `web_browse` 必须 degrade
- **优先级**：中-高（生产化阻塞项；同一进程多 tenant 时安全敏感）

## 17. DLaaS 单进程多 ai_id 部署上限：跨进程 / 跨 GPU 共享 substrate 缺失

- **路径**：
  - 当前 launcher：[`packages/dlaas-platform-launcher/src/dlaas_platform_launcher/instance_manager.py`](../packages/dlaas-platform-launcher/src/dlaas_platform_launcher/instance_manager.py)（每个 ai_id 一个 SessionManager；所有 SessionManager 共享同一个 `OpenWeightResidualRuntime` 实例 = 同一 GPU 同一进程）
  - 共享守门：[`packages/lifeform-service/src/lifeform_service/app.py`](../packages/lifeform-service/src/lifeform_service/app.py)（`_enforce_frozen_for_sharing`）已在 R2 边界校验"shared runtime 必须 frozen"
  - 缺位：跨进程 substrate runtime 共享（IPC / RPC layer）；跨 GPU shard
- **问题**：当前架构下："1 进程 1 substrate runtime + N ai_id" 是上限。这意味着：
  - 单 GPU 容量决定能跑多少并发 ai_id（小模型可能 100+，大模型可能 < 10）
  - 单进程崩溃时全部 ai_id 同时下线
  - 不能用多张 GPU 跑同一 substrate（model parallelism 是 substrate 内部事，但**实例级**水平扩展不行）
  - DLaaS README 没承诺多机部署，但任何 SaaS 化的产品最终需要
- **违反**：不违反任何 R 铁律——单进程模型本身就是当前 substrate 的现实约束。
- **风险**：低（开发期 / 小规模生产可接受），高（中大规模 SaaS 时硬上限）
- **触发条件**：(a) 同一进程并发 ai_id 数 ≥ 50 且单 turn 平均 latency 超过 SLO；(b) 业务方要求 99.9% SLA（单进程崩溃 = 全 fleet 下线，违反）；(c) 单 GPU 显存装不下需要的 substrate 大小
- **推荐修法**：
  1. **第一阶段**（多进程 launcher）：launcher 升级为 controller，每个 ai_id 启动独立子进程跑 `lifeform-service` + 单 SessionManager；launcher 路由 HTTP；substrate runtime 仍是每个子进程一份（不共享，但能水平扩进程）
  2. **第二阶段**（substrate IPC 共享）：`vz-substrate` 加 `RemoteResidualRuntime`，IPC 调本机 substrate server 进程；多 ai_id 进程共享同一 GPU 上的 substrate（避免重复加载模型）
  3. **第三阶段**（多机）：substrate server 跨主机；HTTP / gRPC 协议；与 #16 tool policy 配合做 contract → physical instance 路由
  4. 任何阶段不破 vz-* 内核 0 改动承诺；substrate streaming（#12）和这条 #17 在动 vz-substrate 时应统筹考虑（一次性 additive 改动比分多次 review 成本低）
- **优先级**：低（开发期 / 内部 demo 阶段不阻塞；做 SaaS 时再上）

## 18. Figure F6 PEFT LoRA bake backend 是 stub，真 GPU 训练未接

- **路径**：
  - 接口空壳：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_peft.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_peft.py)（`PEFTLoRABakeBackend.bake(...)` 直接 raise `NotImplementedError("future F6.X packet")`）
  - 当前实际跑的 backend：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_synthetic.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_synthetic.py)（`SyntheticLoRABakeBackend` 用 SHAKE-256 hash 派生 deterministic delta）
  - 公共契约：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_artifact.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_artifact.py)（`LoRABakeBackend` ABC + `FigureLoRAArtifact`）
- **问题**：F6 plan 明文承诺 hybrid 后端：synthetic-first 给 SHADOW + 测试用，PEFT-backed 给真生产用。当前只有 synthetic 真跑，PEFT 是签名 + docstring 钉死的空壳。这意味着：
  - 任何 cite "figure 已能内化语气 / 立场" 的文档只能引用 synthetic 路径，synthetic delta 数学上和原始 corpus 没有 learned 关系（只是 hash 派生的 stable noise，结构上和 substrate adapter 同形但内容不是从训练数据学到的）
  - 上线 1.5B+ Qwen 真训练时会发现 PEFT 接口的 `model_id` / `peft_config` / `runtime_device` / `checkpoint_dir` 字段全是占位 docstring，需要重新 review 一次接口形状
  - 现有 `apply_persona_lora_through_gate` + `PersonaLoRAPool` 接线全部以 synthetic backend 为参照设计；real PEFT 落地时若 adapter 形状（`tuple[float, ...]` 长度 / 数量）变化，pool 与 substrate adapter 的兼容性需要重新验证
- **违反**：不违反 R 铁律。F6 plan 接受 synthetic-first 作为 hybrid 路径之一；PEFT 是边际能力增强，不是必需。
- **风险**：低-中。短期 SHADOW + demo 看不出，长期当 tenant 要"真按 X 人物的语气说话"时 synthetic delta 没有 learned 关系会被识破——synthetic backend 只能保证 layer shape 兼容 + 整体可路由，不能保证 representation drift 真朝 X 的风格走
- **触发条件**：(a) 第一个 tenant / vertical 想从 corpus 真学到 persona representation；(b) GPU 资源就绪 + HuggingFace PEFT 库可装（>= 1× 16GB GPU）；(c) 接 #13 LLM judge 后想跑 "synthetic vs PEFT" 对比 evidence
- **推荐修法**：
  1. 在 `lifeform_domain_figure.lora_bake_peft` 加 `_PEFT_AVAILABLE = importlib.util.find_spec("peft") is not None`，sentinel 后再实例化真训练循环
  2. `PEFTLoRABakeBackend` 加 typed fields `model_id: str` / `peft_config: PEFTLoRAConfig` (新 frozen dataclass) / `runtime_device: Literal["cpu","cuda"]` / `checkpoint_dir: pathlib.Path`；`bake(...)` 用 `peft.LoraConfig` + HF Trainer 跑短 epoch；输出抽 trained adapter weights → `SubstrateDeltaAdapterLayer` tuple；保留 `training_plan_hash` 绑定
  3. 守 R10：bake 完仍要走 `apply_persona_lora_through_gate` 的 OFFLINE gate，不能 bypass；`validation_delta` 与 `capacity_cost` 用真训练 loss / parameter 范数算，而不是默认 0.05 / 0.30
  4. 加 `tests/contracts/test_figure_persona_lora_synthetic_vs_peft_shape.py`：synthetic 与 PEFT backend 的输出 layer 数 / vector_dim / mean_abs_delta 量级应该兼容 pool 的同一 `register(...)` 调用形状（不要求数值一致，只要求 schema 一致）
  5. 单独 packet review，按 `cursor-convergence-workflow.mdc` 走 SHADOW → ACTIVE
- **优先级**：低-中（GPU + corpus license 是前置硬约束，不是软件本身阻塞）

## 19. Figure vertical 默认 corpus 是 reviewer-paraphrased synthetic，archive V2 fetcher + 真 payload 数据集 + parser 三件未做

- **路径**：
  - synthetic 占位语料：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/sample_corpus.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/sample_corpus.py)（`synthetic_einstein_corpus()` 含一个合成 paper / letter / lecture / notebook，明文标注 "synthetic original, not derived from any published primary source"）
  - corpus source adapters：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/)（`ingest_papers.py` / `ingest_letters.py` / `ingest_lectures.py` / `ingest_notebooks.py` —— typed source 转 `IngestionEnvelope`）
  - **archive V1 schema 已就位**：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/archives/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/archives/) —— CPAE / Wikisource / Gutenberg / Internet Archive 4 个 archive 的 typed `*Payload` dataclass + `*_to_paper_source` / `*_to_letter_source` / `*_to_lecture_source` 翻译函数齐全；`ArchiveFetcher` Protocol + `_OfflineArchiveFetcher` 占位明文写"V1 不发 HTTP，V2 packet 接 HTTP fetcher"
  - DLaaS adopt 默认种子：[`packages/lifeform-service/src/lifeform_service/figure_bundle_store.py`](../packages/lifeform-service/src/lifeform_service/figure_bundle_store.py) (`_seed_default_store` 直接调 `synthetic_einstein_corpus()`)
- **问题**：`docs/specs/figure-vertical.md` 承诺 L1 (语气保真) / L3 (引证保真) / L4 (不知拒答) 阶梯靠"全部一手资料"撑起。当前断点分三段：
  1. **没有 V2 HTTPArchiveFetcher** —— archive V1 schema 是"curator 手动喂 pre-downloaded `*Payload`"，4 个 archive 的 `_OfflineArchiveFetcher.fetch(...)` 全部 raise NotImplementedError；任何 URL → bytes 这一步都得手做
  2. **没有 source bytes → cleaned text parser** —— `*Payload.body` 字段假设是已 cleaned 的 plain text，但 Princeton CPAE 是 PDF / facsimile，Wikisource 是 MediaWiki template，Project Gutenberg 是粗糙 HTML，Internet Archive 是 OCR 后 JSON——四种格式各自的 parser / boilerplate-stripper 都不存在
  3. **没有 curated payload 数据集** —— `packages/lifeform-domain-figure/data/` 不存在；没有 `data/cpae/vol{N}-doc{M}.json` 这种预下载 + 反复用的 payload 仓库；想跑真 Einstein 的人现在唯一能做的是手抄进 `CPAEPayload(...)` Python 字面量
  4. 结果：retrieval index / coverage map / style prior 的整体辨识度上限被 corpus 大小拖住——4 个 chunk 的 BM25 + 256-dim hashing embedding 在 in-corpus vs out-of-corpus 边界附近会 noisy（实测 e2e test 必须挑专门的 in-corpus tokens，普通问题命中率低）
  5. 任何"figure 已经能引用 Einstein 论文"的 demo cite 都是把 reviewer paraphrase 当 Einstein 自己的话——和 #14 audience analysis 的"声明 vs evidence"差距同构
  6. asset fetcher (#15) 是 DLaaS 平台层的通用 corpus 抓取模块；figure 的 archive fetcher 与它要么共用要么至少风格统一，目前两条都没接
- **违反**：不违反 R 铁律。corpus 选择是产品 / 法律 / IP 决策，架构上正确：archive payload → typed source → ingestion envelope → retrieval/coverage/style 链路是清的；缺的是 source 来源端 + parser + 实际数据
- **风险**：中。短期 demo / 内部 evidence 没问题（synthetic 文本足够覆盖测试 + 端到端流程），长期要外部 cite 时 synthetic 标注会被发现
- **触发条件**：(a) 第一个 tenant 想用真 figure（活人授权 / 已逝公共领域人物）的 corpus；(b) 对外 demo 时 stakeholder 问 "这是 Einstein 真说过的话吗"；(c) #15 asset fetcher 落地后想统一 figure 与其他 vertical 的 corpus 入口；(d) #23 训练管线 script 落地时发现没有真 payload 数据可跑
- **推荐修法**（按依赖顺序）：
  1. **V2 archive fetcher**：在 `lifeform-domain-figure/corpus/archives/` 下加 `HTTPArchiveFetcher`（CPAE / Wikisource / Gutenberg / Internet Archive 各一个），共用一份 SSRF allowlist + content-type 嗅探（沿用 `lifeform-ingestion` slice 2b 的纪律，与 #15 `AssetFetcher` 接口对齐或共用）
  2. **parser 套件**：`corpus/parsers/` 子模块加 `parse_cpae_pdf` / `parse_wikisource_html` / `parse_gutenberg_html` / `parse_archive_org_ocr_json`；输入 raw bytes + content-type，输出 cleaned plain text（boilerplate stripped），失败 fail-loud
  3. **数据仓库**：在 `packages/lifeform-domain-figure/data/{archive}/` 下放 reviewer 已 curated 的 `*Payload` JSON 序列（先放 10-20 份高优先级 Einstein CPAE 文档作为 minimum viable real corpus），加 `corpus/loaders/load_curated_payloads(archive: str, figure_id: str) -> tuple[*Payload, ...]` loader
  4. `figure_bundle_store._seed_default_store` 加 `corpus_mode: Literal["synthetic", "curated"] = "synthetic"` 参数：synthetic 走当前路径（dev / CI 默认），curated 走 loader → archive translator → bundle compilation；`build_dlaas_app(...)` 入口暴露
  5. **provenance 标记**：加 `figure_corpus_provenance` 字段进 `FigureArtifactBundle`（`Literal["synthetic-placeholder", "curated-primary-source", "scraped-archive-v2"]`），渲染到 grounded decoder 的 evidence pointer + L4 拒答模板里，让用户能区分不同 corpus 来源的 bundle
  6. 守 R8：corpus loader / parser / fetcher 只产 envelope，不直接写 retrieval index / coverage map / style prior；现有 `build_figure_artifact_bundle(FigureBundleInputs(envelopes=...))` 是唯一编译入口
  7. 守 R12：corpus provenance 只读，不反向写 reward / Face；和 #13 / #14 一起在 `tests/contracts/test_dlaas_figure_corpus_no_kernel_writeback.py` 静态守门
- **优先级**：中（产品 / 法律决策是前置；架构上 V1 schema ready，V2 fetcher + parser + 数据三件齐才算 corpus 准备真到位）

## 20. PersonaLoRAPool.activate 是 in-memory passthrough，未接真 GPU 多 LoRA 热插

- **路径**：
  - 当前 pool：[`packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py`](../packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py)（`PersonaLoRAPool.activate(...)` 实现是 `return self.lookup(...)`——只查不动）
  - LoRA artifact 形态：`SubstrateDeltaAdapterLayer` tuple，shape 与 vz-substrate 现有 rare-heavy / online-fast checkpoint 输出兼容
  - DLaaS adopt 接线（缺位）：[`packages/dlaas-platform-launcher/src/dlaas_platform_launcher/instance_manager.py`](../packages/dlaas-platform-launcher/src/dlaas_platform_launcher/instance_manager.py) 现在不调 `pool.activate(...)`，也没在 `OpenWeightResidualRuntime` 上下文里把激活的 LoRA 推到模型权重
- **问题**：F6 plan 写 "PersonaLoRAPool 同一冻结基底上热插 N 个 LoRA"。当前实现只完成"N 个 LoRA 在内存"，"热插"那一半（即真把选中的 adapter delta 加到 GPU-resident frozen base 的 forward 上）只在 docstring 里描述为 future S-LoRA / vLLM multi-LoRA 兑现。意味着：
  - 即便 `apply_persona_lora_through_gate` 走完 OFFLINE gate + 把 artifact 注册进 pool，runtime 实际生成时**还是用裸 frozen base**（没有 LoRA delta 影响 forward）
  - 当前 pool 的价值是"artifact 寿命管理 + 跨 ai_id 隔离 + 审计 record id"——这是必要的 prerequisite，但不是 F6 plan 完整意图
  - 与 #17 cross-process / cross-GPU substrate 共享强相关：那条债的"第二阶段 RemoteResidualRuntime"和这条的"真热插"应该一次设计完
- **违反**：不违反 R 铁律。Plan 明文允许"接口与 S-LoRA / vLLM multi-LoRA 兑现留 docstring"
- **风险**：中。短期 SHADOW + demo 看不到差别（synthetic LoRA delta 反正也不影响真行为），长期接 PEFT (#18) 后会发现 baked LoRA 没生效——产品上"两个 ai_id 在同一进程跑出不同 persona"的承诺要靠这一段才能落地
- **触发条件**：(a) #18 PEFT backend 落地后第一次想看 baked LoRA 真改变 substrate 输出；(b) 同一进程并发 ≥ 2 个 ai_id 且各自有不同 persona LoRA；(c) #17 cross-GPU 共享 substrate 的设计开始
- **推荐修法**：
  1. `vz-substrate` 加 `LoRAAwareResidualRuntime` Protocol：`activate_lora(layers: tuple[SubstrateDeltaAdapterLayer, ...]) -> contextlib.AbstractContextManager`；上下文进出时把 layers 加到 / 撤出 frozen base 的对应 attention block
  2. `OpenWeightResidualRuntime` 与 `TransformersOpenWeightResidualRuntime` 实现该 Protocol（小模型直接 monkey-patch forward；大模型走 vLLM multi-LoRA / S-LoRA 路径）
  3. `PersonaLoRAPool.activate(...)` 改为：(a) 查 record；(b) 调上游 runtime 的 `activate_lora(record.adapter_layers)`；(c) return AsyncContextManager that auto-deactivates on exit
  4. `dlaas-platform-launcher.InstanceManager` 在 `acquire(ai_id, ...)` 时把 ai_id → figure_id 映射加进 SessionManager；session.run_turn(...) 进入前 `with pool.activate(figure_id):` 包一层
  5. 与 #17 第二阶段（substrate IPC 共享）统筹设计；与 #16 tool_policy_snapshot 接线类似，都是 launcher 把 contract / ai_id 维度的策略推到 runtime
  6. 守 R2：activate 改的是 controller 层 adapter delta，frozen base 不动；测试套加 "activate 前后 base model state_dict hash 不变"
- **优先级**：中（与 #17 / #18 强耦合；单独做收益小，统筹做一次性 review 成本低）

## 21. F5 Steering bake 在 hashing-embedding 坐标系，未在 substrate 真残差流上提取方向

- **路径**：
  - 当前 bake：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_bake.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_bake.py)（`_direction_for_pairs` 算 `weighted_mean(positive - negative)` → unit-norm，全在 256-dim hashing embedding 坐标系里）
  - 数据 prep：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_data_prep.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_data_prep.py)（`_pair_to_training_pair` 用 `lifeform_domain_figure.retrieval_index._hashing_embedding`——和检索同坐标系，是有意为之）
  - 缺位：`vz-substrate` 真 residual stream 抽取 + LDA / CCS-style readout 直接在 model hidden states 上学方向
- **问题**：plan 写 "real CPU backend"——当前实现是 mathematically real CPU contrastive readout（带 reviewer confidence weighting + cosine margin scale），但坐标系是 hashing embedding 而不是真 substrate 残差流。这意味着：
  - 拿到的 steering vector 在 hashing embedding 空间里有定向意义（positive paraphrase vs negative paraphrase 真分开），**但**这个空间和 substrate 内部 residual stream 之间没有学到的映射关系
  - `to_substrate_adapter_layers(...)` 把 vector 包成 `SubstrateDeltaAdapterLayer` 时，layer.delta_vector 和 substrate 真实 residual stream 的对齐关系靠 `vector_dim` 的隐式假设；当 #20 真 hot-swap 路径上线时，如果 substrate 的实际残差维度不是 256，这条路径会失配
  - 真正"按 Einstein vs Bohr 倾向"的表征应该从 substrate 在 positive / negative paraphrase 上的真 hidden states 抽差异，而不是从 reviewer text 的 hashing 抽差异——这是 contrastive activation steering 的标准做法（Anthropic CAA / RepE 等论文）
- **违反**：不违反 R 铁律。Plan 接受这条路径作为 hybrid 后端的 "CPU 可跑"分支；只是把"steering 真生效"门槛设在了"plus #20 hot-swap"+"plus #18 真 PEFT"两个前置后
- **风险**：中。短期 SHADOW 看不出（synthetic LoRA + 不真热插，所以 steering vector 也只是 schema 占位），长期当 #18 + #20 都 ready 时会发现 steering layer 的 delta 量级 / 方向都不真起作用
- **触发条件**：(a) #18 PEFT backend + #20 真热插同时 ready；(b) 想跑 contrastive steering vs LoRA 的对比 evidence；(c) 想 cite "figure 立场已被 steering 校准"
- **推荐修法**：
  1. 在 `vz-substrate` 加 `extract_residual_for_text(text: str, layer_indices: tuple[int, ...]) -> dict[int, tuple[float, ...]]` additive 接口（与 #12 streaming / #20 hot-swap 一起 review）
  2. 在 `lifeform_domain_figure.steering_data_prep` 加 `RealResidualSteeringDataPrep` 派生：`build_steering_training_plan(..., residual_extractor: ResidualExtractor)`；`_pair_to_training_pair` 改走 `extractor.extract(pair.figure_stance)` 而不是 `_hashing_embedding(...)`
  3. 在 `lifeform_domain_figure.steering_bake` 加 `RealResidualSteeringBakeBackend` 派生（沿用 `bake_steering_set` 的 contrastive readout 算法，但坐标系是真 residual）；保留 hashing-embedding 路径作为 SHADOW + 测试 fallback
  4. 守 R8：steering bake 输出仍是 `FigureSteeringSet` + `SubstrateDeltaAdapterLayer` tuple；上下游接口不变
  5. 加 `tests/contracts/test_steering_real_vs_hashing_shape.py`：两条路径输出的 set 必须可互换 plug 进 `attach_baked_steering(...)`（schema 一致），不要求数值一致
- **优先级**：低-中（前置依赖 #18 + #20）

## 22. DLaaS adopt 主路径未自动 hook `lookup_figure_bundle` + `register_bundle_persona_lora`

- **路径**：
  - 助手函数已就位：[`packages/lifeform-service/src/lifeform_service/figure_bundle_store.py`](../packages/lifeform-service/src/lifeform_service/figure_bundle_store.py)（`lookup_bundle(default=None, *, bundle_id=...)` 与 `register_bundle_persona_lora(bundle, *, pool=None)`）
  - 公开 surface：[`packages/lifeform-service/src/lifeform_service/__init__.py`](../packages/lifeform-service/src/lifeform_service/__init__.py)（两个 helper 都已 re-export 为 `lookup_figure_bundle` / `register_bundle_persona_lora`）
  - **缺位的调用点**：DLaaS adopt route handler（应该在 `dlaas-platform-api` 的 adopt 路径或 `dlaas-platform-launcher.InstanceManager.acquire(...)` 入口）调 `lookup_figure_bundle(bundle_id=template.figure_artifact_id)` + `register_bundle_persona_lora(bundle)`，但当前 grep 显示这两个 helper 只在 [`tests/test_einstein_vertical_smoke.py`](../packages/lifeform-service/tests/test_einstein_vertical_smoke.py) 与本身的 docstring / 公开 surface 里出现，DLaaS 主路径上没人调
  - vertical 自带的 bundle 注入路径：[`packages/lifeform-service/src/lifeform_service/verticals.py`](../packages/lifeform-service/src/lifeform_service/verticals.py) `_try_einstein` 工厂直接构造 bundle + 注入 synthesizer，不走 figure_bundle_store
- **问题**：F4.2 + F6.3 plan 承诺 DLaaS adopt 应该：(a) 读 `template.figure_artifact_id`；(b) `lookup_figure_bundle` 拿 bundle；(c) 注入到 LifeformLLMResponseSynthesizer（已实现）；(d) `register_bundle_persona_lora(bundle)` 把 bundle.lora 推进 PersonaLoRAPool（**未实现**）。当前现实：
  - vertical Einstein 工厂自己构造 bundle，不读 template.figure_artifact_id；template 加这字段只是 schema 占位，没 wiring 真消费
  - bundle.lora 即便走完 `apply_persona_lora_through_gate`，DLaaS adopt 也不会自动登记到 PersonaLoRAPool；要靠测试 / 显式脚本调用
  - 任何 cite "DLaaS adopt 已能加载 figure persona LoRA" 的文档都是把"接口齐了"当成"线接通了"
- **违反**：不违反 R 铁律。R8 / R15 / R10 都允许这层是手动 wiring；只是 plan 的 P4.2 + P6.3 完成度被高估
- **风险**：低-中。短期 SHADOW + e2e 测试都用直接 invocation 路径，看不到差别；长期 tenant 上 figure_artifact_id 后会发现 template 字段不生效——和 #16 tool_policy_snapshot 同构（持久化了但运行时不读）
- **触发条件**：(a) 第一个 tenant 在 template 里设 `figure_artifact_id` 期望 adopt 自动加载；(b) 第一个 tenant 想看到 baked persona LoRA 在 chat 时真生效；(c) 接 #15 asset fetcher 后 figure corpus 产 bundle → bundle 进 store → adopt 自动 hook 这条链需要闭合
- **推荐修法**：
  1. 在 `dlaas-platform-launcher.InstanceManager.acquire(...)` 或 `dlaas-platform-api.control_plane._handle_adopt(...)` 加：
     - `if final_template.figure_artifact_id: bundle = lookup_figure_bundle(bundle_id=final_template.figure_artifact_id)`
     - `synthesizer.with_figure_bundle(bundle)`（已支持）
     - `if bundle is not None: register_bundle_persona_lora(bundle)`（pool 默认走进程级 `default_persona_lora_pool()`）
  2. 与 #20 真 hot-swap 一起设计：register 完后 `pool.activate(figure_id)` 进 SessionManager 上下文，否则注册了也没生效
  3. 与 #16 tool_policy_snapshot 接线统筹：都是 "template.X 字段在运行时被消费" 的同构问题，可以一次性补全
  4. 加 `tests/service/test_dlaas_adopt_loads_figure_bundle.py`：template 含 figure_artifact_id → adopt → instance 的 synthesizer.figure_bundle 非 None；template 含的 bundle.lora 非 None → 默认 pool.has(figure_id) 为 True
  5. 守 R8：adopt 只调 helper 公开 surface，不直接 import figure-vertical 内部模块
- **优先级**：低-中（与 #16 / #20 同时做更高效）

## 23. Figure vertical 训练管线 script 完全没有，所有 bake / gate apply 只是 Python 函数

- **路径**：
  - 现有 Python 函数（无 CLI 包装）：
    - `build_figure_artifact_bundle(FigureBundleInputs(...))` ([`packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py))
    - `bake_steering_set(plan)` + `apply_steering_through_gate(...)` ([`packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_bake.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_bake.py))
    - `SyntheticLoRABakeBackend().bake(plan)` + `apply_persona_lora_through_gate(...)` ([`packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_synthetic.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_synthetic.py) + [`gate_apply.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/gate_apply.py))
    - `register_bundle_persona_lora(bundle)` ([`packages/lifeform-service/src/lifeform_service/figure_bundle_store.py`](../packages/lifeform-service/src/lifeform_service/figure_bundle_store.py))
  - **缺位的脚本**：`scripts/` / `examples/` 下没有 `figure` / `einstein` / `lora` / `steering` / `bake` 任何 CLI 入口；grep 结果空
  - 唯一调动 bake / gate apply 的真实代码：[`packages/lifeform-domain-figure/tests/`](../packages/lifeform-domain-figure/tests/)（test fixture，不是产线 script）
- **问题**：F5 / F6 plan 设计 OFFLINE-gated 训练 + 部署流程，但当前所有 bake / gate apply / pool register 步骤都只能在 Python REPL 或测试里跑。结果：
  - 即便用户准备好真 Einstein corpus（前置 #19）、真 PEFT backend（前置 #18），也没有一个 one-shot CLI 跑完 `fetch corpus → ingest → bundle → steering bake → lora bake → gate apply → register in pool → save bundle to disk`
  - 操作员要"重新 bake Einstein 的 LoRA"必须手写 Python；没有可记录、可 review、可 rerun 的 audit trail
  - rollback 流程没有 CLI surface：`apply_*_through_gate` 返回 `previous_record_id` / `previous_artifact_id`，但没有 `--rollback-to=<id>` 这种操作员入口
  - bundle 不会 persist 到 disk —— 当前 `FigureArtifactBundle` 只在内存；`figure_bundle_store` 是 process-level in-memory dict（重启进程就丢）；没有 `save_bundle(path)` / `load_bundle(path)` 也没有"bundle 仓库"概念
  - 任何 evidence run 都得自己写 80-150 行 Python boilerplate 串起调用，跑出来的 artifact 在哪、用什么 corpus、用了哪个 backend、谁 review 过的——全凭 commit message
- **违反**：不违反 R 铁律。F5 / F6 plan 接受"先把函数做对，CLI 是 ops 后续工作"；只是把"figure vertical 已可用"的解释门槛设在了"加上 #23 + #19 + #18 + #20 + #22"
- **风险**：低-中。短期开发期 / 测试 fixture 跑得通，长期产品化时这是最贴近用户操作的一层，缺了等于 ops team 接不了手
- **触发条件**：(a) 第一次想把 baked figure bundle 跑给非工程师 review；(b) #19 真 corpus 数据集落地后想跑端到端；(c) #18 PEFT backend 落地后想 audit "什么 plan + 什么 corpus + 什么 backend → 出了什么 bundle"；(d) 任何"重 bake → 走 OFFLINE gate → 替换 active artifact"流程需要可重复跑
- **推荐修法**（按 minimum viable 顺序）：
  1. **bundle 持久化**：`lifeform_domain_figure` 加 `bundle_io.py` —— `save_figure_bundle(bundle: FigureArtifactBundle, dir: pathlib.Path) -> pathlib.Path` / `load_figure_bundle(dir: pathlib.Path) -> FigureArtifactBundle`；用 frozen pickle 或 typed JSON + 整数化 hash 校验；落到 `data/bundles/{figure_id}/{bundle_id}/` 目录
  2. **bake CLI**：`scripts/figure_bake.py`（或 `python -m lifeform_domain_figure.cli`）三个子命令：
     - `bake-bundle --figure einstein --corpus-mode {synthetic,curated} --out data/bundles/einstein/`
     - `bake-steering --figure einstein --bundle <bundle_id> --gate offline --rollback-evidence "<text>" --out data/bundles/einstein/{new_id}/`
     - `bake-lora --figure einstein --bundle <bundle_id> --backend {synthetic,peft} --gate offline --rollback-evidence "<text>" --out data/bundles/einstein/{new_id}/`
  3. **gate audit**：每条 bake 子命令完成后写 `data/audit/{timestamp}_{action}_{figure}.json`，含 `corpus_provenance` / `backend_id` / `validation_delta` / `capacity_cost` / `rollback_evidence` / `gate_decision` / `block_reasons`；和 `vz-cognition` 的 `RuntimeAdaptationAudit` 风格对齐
  4. **rollback CLI**：`figure_bake rollback --figure einstein --to-bundle <previous_id>` —— 调 `figure_bundle_store.register(previous_bundle)` + `pool.deregister(current_record_id)` + `pool.register(...previous artifact...)` + 写 audit 记录
  5. **一键端到端 demo**：`scripts/figure_demo_einstein.sh`（或 Python） —— synthetic corpus 路径下 bundle → steering → lora → register → 跑一句话 chat → 输出引证 + L4 refusal demo；和 `scripts/run_eq_evidence_bundle.sh` (#10B item 3 那条) 风格对齐
  6. **守 R10 / R15**：CLI 不能 bypass `apply_*_through_gate`；任何 `--bake-and-apply` 都必须把 EvaluationSnapshot 路径走通（接受 `--evaluation-snapshot path/to/snapshot.json` 输入）；每次 apply 出 `applied / record_id / previous_record_id` 进 audit
  7. 加 `tests/service/test_figure_bake_cli_smoke.py`：以 synthetic corpus + synthetic LoRA 跑过整条 CLI；audit 文件结构正确；rollback CLI 真换出 bundle
- **优先级**：中（独立可做，不强依赖 #18 / #19；做完后这三件 + #22 一起把 figure vertical 从"函数齐了"推到"可以交给 ops"）

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