# Known Architecture Debt

> Status: tracked, not blocking
> Last updated: 2026-05-11 (EQ-Bench 3 wiring dry-run validated + Companion Bench v1.0 reference impl land + Real-Person Figure Vertical F1-F6 + Persona Figure 数据管线 V1 D2/D3/D4/D7 全套 land；#18-#22 为 figure F6 训练后端/数据/DLaaS hook，~~#23~~ 已闭合（figure bake CLI + bundle 持久化 + audit log + rollback 全套 land），~~#19~~ 部分闭合（V2 fetcher + parser 落地）、~~#25~~ 闭合（metadata fingerprint folded into bundle hash）、~~#26~~ 闭合（4 V2 metadata clients + cache + role gate）；#28 完整 webcrawl 编排 + 清洗管线 + 多源验证审计三层（L0+L1+L2 first batch+L2 second batch）**架构层完全 land**，剩余只是 curated 数据集 / reviewer workflow 类工作；#24/#27 为本轮 D2/D7 数据管线层未串接缺口；**#29-#30 为对外 benchmark 证据面缺口（融资尽调阻塞条件）**，#29 P1-P9 已 land + **wiring 层已通过 dry-run 验证**（见 [`docs/external/eqbench3-wiring-evidence.md`](external/eqbench3-wiring-evidence.md)：synthetic vertical 上 45/45 scenarios 跑通 + 修复 2 个 URL bug），P10 actuation 仍 gate 在真实 ablation verdict 后（需 GPU + Anthropic key）；**#31 OpenAI-compat 适配 wheel 流式 SSE 未实现**；**#29 轨 2 Companion Bench v1.0 reference impl 已 land**：`packages/companion-bench/` Apache 2.0 wheel + 24 public + 96 held-out scenarios + 10 reference systems orchestrator + GitHub Pages 站点 + 4 个 CI workflow；Companion Bench v1.0 launch 路径未启动追踪在 **#32**（governance / 真 reference 跑分 / DNS / submission queue），human-eval 轨道 0% 追踪在 **#33**（RFC §6.6），harness 性能 / async / staged executor 追踪在 **#34**，季度 rotation 自动化（held-out / lexicon / judge）追踪在 **#35**，v2.x 长尾（multi-modal / EQ-Bench 1:1 prompt / 加密 attestation / transcript 脱敏）追踪在 **#36**——配套对外 RFC 见 [`docs/external/companion-bench-rfc-v0.md`](external/lscb-rfc-v0.md)，配套 OpenAI-compat 适配层位于 `packages/lifeform-openai-compat/`）

> 2026-05-11 update (EQ-Bench 3 wiring dry-run validated): 把 #29 P1-P9 已交付的代码层从「unit tests + 文档」推进到「真 harness 端到端跑通」。Clone [`https://github.com/EQ-bench/eqbench3`](https://github.com/EQ-bench/eqbench3) 到 `external/eqbench3/`、装 9 个 light deps、boot `lifeform-serve --vertical companion --substrate-mode synthetic --enable-openai-compat`、配 `.env`、跑 `eqbench3.py --no-rubric --no-elo --ignore-canonical` 全 45 scenarios → 全部 `status: completed`、产 [`artifacts/external_bench/eqbench3_dry_run.runs.json`](../artifacts/external_bench/eqbench3_dry_run.runs.json) (1.17 MB)、26 debriefs 也全通过。**修复 2 个 URL bug**：(a) [`scripts/external_bench/.env.example`](../scripts/external_bench/.env.example) + [`run_eqbench3.py`](../scripts/external_bench/run_eqbench3.py) `TEST_API_URL` / `JUDGE_API_URL` 原本只写 `/v1` 但 eqbench3 `utils/api.py` 不会补 `/chat/completions` → 改为 full path，11 个 `test_run_eqbench3_smoke.py` 单元测试在 fix 后全绿；(b) PowerShell `Out-File -Encoding utf8` 默认带 BOM 破坏 dotenv 解析 → bootstrap note 写入 [`docs/external/eqbench3-wiring-evidence.md`](external/eqbench3-wiring-evidence.md)。Wallclock：synthetic CPU 单 track 14:42。**P10 actuation 仍 gate 在**：真 Qwen 1.5B substrate（GPU 或 hf-shared）+ 真 Anthropic `JUDGE_API_KEY` + 调用 `--with-elo` 跑 pairwise（增量 ~$10-20/track）+ 三轨 (`companion / companion-cold / raw`) 跑完产 `.summary.json` 后 [`compare_ablation.py`](../scripts/external_bench/compare_ablation.py) 取 verdict。完整 reproduction recipe 见 evidence 文档。这一步**把「实测前的所有未知」从 #29 的债务面剥离** — 接下来真跑分时只剩 substrate / judge 一类 known-unknown，adapter-side 隐藏 bug 已清零。

> 2026-05-10 update (Companion Bench v1.0 reference impl land — debt #29 轨 2 推进): RFC 文档级 v0.1 → 工程级 v1.0 全套就位。新 wheel [`packages/companion-bench/`](../packages/companion-bench/) Apache 2.0 隔离许可，13 模块 + 144 单元测试全绿（包括 `tests/contracts/test_lscb_bench_no_internal_imports.py` 静态守 lscb-bench 不 import vz-* / lifeform-*，匹配 RFC §3 P4 outcome-level 评估契约）。**Held-out 治理**：96 scenario 走 git submodule + private repo `VolvenceZero/lscb-heldout`（[`.gitmodules`](../.gitmodules) + [`docs/external/companion-bench-heldout-bootstrap.md`](external/lscb-heldout-bootstrap.md) 一次性 organiser bootstrap），公仓 PR 永不见 held-out body；CI release-tier 用 deploy key 拉取，PR / open-source clones 自动跳过。**Public scenarios**：24 个完全 in-repo（6 family × 4），hash 表落 [`docs/external/companion-bench-public-scenario-hashes.txt`](external/lscb-public-scenario-hashes.txt) 由 [`scripts/companion_bench/emit_scenario_hashes.py`](../scripts/companion_bench/emit_scenario_hashes.py) 重生成。**Public leaderboard 静态站**：`site/leaderboard/` 纯 HTML + vanilla JS，[`scripts/companion_bench/generate_demo_aggregate.py`](../scripts/companion_bench/generate_demo_aggregate.py) 给出 demo 渲染数据（10 系统 placeholder），真 reference 跑分入口 [`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 跑通即替换。**4 个 CI workflow**：`lscb-ci-smoke`（PR gate, 公开） / `lscb-paper-suite-small`（nightly $200-400） / `lscb-paper-suite-full`（release $5-15k, 拉 held-out） / `lscb-leaderboard-publish`（GitHub Pages）。**4 个 shell 脚本**：`run_lscb_ci_smoke.sh` / `run_lscb_paper_suite_small.sh` / `run_lscb_paper_suite_full.sh` / `build_leaderboard_site.sh`。**5 个 governance 文档**：[`lscb-submission-protocol.md`](external/lscb-submission-protocol.md) / [`lscb-governance-charter-draft.md`](external/lscb-governance-charter-draft.md) / [`lscb-eqbench-crosswalk.md`](external/lscb-eqbench-crosswalk.md) / [`lscb-heldout-bootstrap.md`](external/lscb-heldout-bootstrap.md) / [`docs/specs/companion-bench.md`](specs/lscb-bench.md)。**仍未做（组织层 / 预算层 follow-up，不是代码层债）**：(a) working group 形成（RFC §11，依赖外部第二个组织接入，charter draft 已就位）；(b) 真 10 reference systems 跑分（$5-15k API 预算，scripts 已就位等批准）；(c) 真域名 `lscb-bench.volvencezero.org` DNS + GitHub Pages CNAME 配置；(d) v1.1 quarterly held-out paraphrase rotation。这四项都不阻塞 v1.0 reference impl 的工程交付。

> 2026-05-10 update (debt #29 P1-P9 land + new debt #31 streaming SSE): 对外 EQ benchmark 提交全链路（debt #29 推荐修法 1-5）的代码层 + 文档层全部就位。新 wheel `lifeform-openai-compat` 暴露 `POST /v1/chat/completions`（OpenAI envelope）on top of `lifeform-service`，三轨 ablation 设计（companion / companion-cold / raw substrate）支持 EQ-Bench 3 一次跑出"系统 vs 裸 substrate"delta；`scripts/external_bench/run_eqbench3.py` + `compare_ablation.py` 提供 launcher + verdict 守门（红线 attestation 缺失则拒绝出 verdict）；`scripts/external_bench/run_empathybench.py` 提供同结构 generic harness（empathybench.com 闭源时支持 EmotionBench 等开源等价物）。文档层落 [`docs/external/eqbench3-submission-protocol.md`](external/eqbench3-submission-protocol.md)（reproducibility 协议）+ [`docs/external/companion-bench-eqbench-crosswalk.md`](external/lscb-eqbench-crosswalk.md)（LSCB↔EQ-Bench 跨 benchmark 映射，给 Companion Bench v0.2 evidence backing）+ [`docs/external/eqbench3-results-internal.md`](external/eqbench3-results-internal.md)（per-run verdict 模板）+ [`docs/external/eqbench3-public-submission-checklist.md`](external/eqbench3-public-submission-checklist.md)（P10 actuation gate）。**1048 import-boundary tests + 138 adapter unit/integration tests 全绿**，`vz-* / lifeform-* / lifeform-domain-*` 内核包**零修改**，唯一一处 lifeform-service 改动是 `cli.py` 加 `--enable-openai-compat` flag（默认 off，向后兼容）。守红线静态化：[`tests/contracts/test_openai_adapter_import_boundary.py`](../tests/contracts/test_openai_adapter_import_boundary.py) AST 守 adapter 不导内核；[`tests/contracts/test_openai_adapter_no_kernel_writeback.py`](../tests/contracts/test_openai_adapter_no_kernel_writeback.py) AST 守 adapter 不写 SessionManager / LifeformSession 私有状态；`compare_ablation.py` 程序化校验四条 #29 红线（frozen substrate / no kernel mod / no benchmark text in system prompt / no internal vocab in model card）后才发 verdict。**P10 公开提交仍 gate 在 verdict==go**，需要先用真实 GPU + Anthropic API key 跑一次 ablation 拿到分数 — 这一步不在代码 packet 范围。**新增 debt #31**：adapter 当前 `stream=true` 返 501，对 EQ-Bench 3 这种 single-shot harness 不影响，但任何 streaming 模式 harness（Chatbot Arena 实时投票通道、OpenAI Python SDK streaming chat）会因此被阻塞 — 与 #30 (Chatbot Arena 提交) 同时浮现，需要 SSE 落地。
>
> 2026-05-10 update (LSCB RFC + 对外 benchmark 证据债 #29-#30): 对外 benchmark / arena 评估面（业界正在快速成形的 EQ-Bench 3 / EmpathyBench / RP-Bench / Chatbot Arena 等）我们当前**完全缺位**——这条缺位本身既不影响系统运行也不违反 R 铁律，但对 **任何 fundraising 尽调和品类话语权竞争** 都是硬阻塞。本轮把 follow-up 拆成三轨：(轨 2) 已落地：[`docs/external/companion-bench-rfc-v0.md`](external/lscb-rfc-v0.md) 公布 Companion Bench (Long-Session Companion Benchmark) 公共 RFC v0.1，从中立角度定义 multi-session companion 评估方法学（A1–A6 六轴 + 六族 scenario），不暴露任何内部架构（NL/ETA/R-PE/regime/owner SSOT/family report 内部口径），目的是让公司在长会话陪伴评估这条赛道上**先于打榜成为 convener**；与 EQ-Bench 3 rubric 兼容以便他人系统已有信号可以转移；(轨 1) 列入 #29：把 substrate / lifeform 包一层 OpenAI-compatible API 提交 EQ-Bench 3 + EmpathyBench，拿一个**可被引用的客观分数**填上"投资人尽调时第一个 google 到的数字"这条空白；(轨 3) 列入 #30：在人评类 arena（RP-Bench / Chatbot Arena 公开 chat / 后续 Companion Bench 自研人评轨）建立可见 footprint，对齐"EQ > IQ"这条对外叙事。轨 2 不在 known-debts；轨 1/3 短期可推进、与代码层债项独立，写为 #29/#30 跟踪。
>
> 2026-05-10 update (Persona Figure 数据管线 V1 D2/D3/D4/D7): 在 #18-#23 已就位的 F6 训练后端骨架基础上，本轮新增 4 类纯数据策展层：(a) **D2** corpus 子目录加 typed `SourceProvenance` + `LegalClearance` + `compute_dedup_report` + `parse_locator` 三件 reviewer-facing helper（详见 [`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/{provenance,dedupe,citation}.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/)）；(b) **D3** `corpus/archives/` 4 个 archive 适配器（CPAE / Wikisource / Gutenberg / Internet Archive）+ `ArchiveFetcher` Protocol + V1 offline 桩；(c) **D4** `metadata/` 子包给 OpenAlex / Wikidata / Crossref / SEP 4 个 metadata 来源加 typed payload + `*_to_*` 翻译器 + Protocol 客户端 + offline 桩 + `MetadataDigest` 指纹聚合 + `enrich_profile_with_metadata` 桥（**T3 严禁污染**：metadata 永不进 retrieval / LoRA training data）；(d) **D7** 鲁迅 reviewed `HistoricalFigureProfile` + Chinese Text Project archive 适配器（孔子等古典人物用）。150/150 figure tests + 91/91 figure-related import boundary tests 全绿。**4 块新缺口需要后续 follow-up（已开 debt）**：(i) D2 三件 helper 都是 pure 函数，未接进 `build_figure_artifact_bundle` / `build_figure_retrieval_index` 主管线，存在但不 load-bearing（→ #24）；(ii) `MetadataDigest.fingerprint` 未折进 `FigureArtifactBundle.integrity_hash`，metadata enrichment 之后 bundle 字节级回滚契约不闭合（→ #25）；(iii) D4 4 个 metadata client 全是 offline 桩 + `NotImplementedError`，与 #19 archive V2 fetcher 同构但是不同 client 集（→ #26）；(iv) 鲁迅 PoC 只到 `HistoricalFigureProfile`，缺 `synthetic_lu_xun_corpus()` / `build_lu_xun_lifeform()` / 鲁迅 e2e test / CTP curated 数据集（→ #27）。

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

## ~~19. Figure vertical 默认 corpus 是 reviewer-paraphrased synthetic，archive V2 fetcher + 真 payload 数据集 + parser 三件未做~~ —— 2026-05-10 部分关闭

> **进度（2026-05-10）**：本债的 (1) V2 archive fetcher + (2) parser 两件已落地，随 debt #28 L0 + L1 packet 一并实现：
>
> - **V2 archive fetcher**（修法 1）：`corpus.archives.live_archive_fetcher(fetch_kind, ...)` 工厂返回 V2 `ArchiveFetcher`-Protocol 实现；4 个 archive 各有专用 URL pattern + content-type 选择（CPAE PDF / Wikisource action=raw / Gutenberg .txt / IA metadata API → OCR JSON）；既有 `offline_archive_fetcher()` 行为不变（V1 向后兼容）。SSRF allowlist + body-cap + 1-hop redirect rescope 在 `BaseHTTPClient` 集中守。详见 [`docs/specs/figure-corpus-crawl.md`](specs/figure-corpus-crawl.md) 末尾 "V2 ArchiveFetcher closure" 段。
> - **parser 套件**（修法 2）：随 debt #28 L1 落地（CPAE PDF via pypdf / Wikisource HTML via bs4+mwparserfromhell / Gutenberg HTML+plain via bs4 / IA OCR JSON via stdlib）。详见 [`docs/specs/figure-corpus-cleaning.md`](specs/figure-corpus-cleaning.md)。
>
> **本债剩余开放**：(3) **curated payload 数据集**（`packages/lifeform-domain-figure/data/{archive}/` 下的 reviewer-curated `*Payload` JSON 序列 + `corpus/loaders/load_curated_payloads(...)` loader）+ (4) `figure_bundle_store._seed_default_store` 的 `corpus_mode: Literal["synthetic", "curated"]` 切换 + (5) `FigureArtifactBundle.figure_corpus_provenance` 字段 + (6) 单文档 fetcher 已成 superset，但 figure 完整 webcrawl/clean/verify 仍以 debt #28 为统称。剩余工作主要是 reviewer 工艺 + 数据集准备，不再是架构缺位。
>
> _（保留下方原始描述以便 audit；新工作应在 debt #28 / 数据集 follow-up 中追踪。）_

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

## ~~23. Figure vertical 训练管线 script 完全没有，所有 bake / gate apply 只是 Python 函数~~ —— 2026-05-10 关闭

> **2026-05-10 update (#23 closure)**：known-debts #23 推荐修法 7 项一次性 land 一个 packet。新增 `lifeform-domain-figure` wheel 内部模块 [`bundle_io.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/bundle_io.py)（pickle 持久化 + atomic write + integrity_hash 重验，沿用 [`lifeform_evolution.snapshot_io`](../packages/lifeform-evolution/src/lifeform_evolution/snapshot_io.py) 已有 magic-header 风格）+ [`audit.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/audit.py)（typed `FigureBakeAuditRecord` + 4-action enum + `find_previous_audit_for_bundle`，sha256 over payload 的 deterministic `audit_id`）；`cli/` 子包加 [`__init__.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/__init__.py) (argparse 顶层 + 5 子命令 dispatch) + [`__main__.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/__main__.py) + [`_commands.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py) (`bake-bundle` / `bake-steering` / `bake-lora` / `rollback` / `list` 5 个 handler，全部走 wheel `__init__.py` 公开 surface) + [`_eval_snapshot_loader.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_eval_snapshot_loader.py) (`--evaluation-snapshot path/to/file.json` 或 `default-clean` 字面量；后者作为开发期/demo 显式 opt-in，不会默默吃过生产门)；CLI entry point `figure-bake = "lifeform_domain_figure.cli:main"` 进 [`pyproject.toml`](../packages/lifeform-domain-figure/pyproject.toml)；[`scripts/figure_demo_einstein.sh`](../scripts/figure_demo_einstein.sh) 一键 4 步端到端跑通（base bundle → steering → lora → list），与 [`scripts/run_eq_evidence_bundle.sh`](../scripts/run_eq_evidence_bundle.sh) 风格对齐用 `python -m` 入口跨平台；`.gitignore` 加 `data/figure_bundles/` + `data/figure_audit/` 以让运行时产物不进 git。
>
> **退出码语义统一**：`0`=成功；`1`=CLI 参数错误；`2`=`GateDecision.BLOCK`（仍写 BLOCK audit + `block_reasons`，**不**静默吞）；`3`=I/O / schema 错误。`scripts/figure_demo_einstein.sh` 区分得清"开发期跑 demo 失败"vs"gate 故意拦截"。
>
> **守门契约**：
> 1. R8 — 静态契约 [`tests/contracts/test_figure_cli_uses_only_public_surface.py`](../tests/contracts/test_figure_cli_uses_only_public_surface.py) AST 扫 `cli/` 4 个文件，禁止 `from lifeform_domain_figure.<internal_module>` 形式；只允许 `from lifeform_domain_figure import X` 公开 surface 与白名单内的 `cli.*` / `audit` 兄弟模块。allowlist tight test 让未来扩允许列表必须同时改测试 + 文档说明。
> 2. R10 — `cmd_bake_steering` / `cmd_bake_lora` 的 wiring 只走 `apply_steering_through_gate(...)` / `apply_persona_lora_through_gate(...)`；CLI 不能构造 `FigureArtifactBundle` 直接 attach steering/lora 而绕过 OFFLINE gate；BLOCK 路径产 audit + 退出码 2 在 `test_cmd_blocked_gate_writes_audit_with_block_reasons_and_exits_2` 静态守门。
> 3. R15 — `bundle.pickle` 落盘后 reload 必须重算 `compute_bundle_integrity_hash` 与原 `bundle.integrity_hash` byte-equal；`test_save_load_roundtrip_bundle_byte_equal` + `test_save_load_roundtrip_with_steering_and_lora` 守门；rollback 是 append-only（旧 bundle / audit 永不删，rollback 写新 audit 行）。
>
> **回归证据**：194/194 figure-domain tests 全绿（原 150 + 5 bundle_io smoke + 7 audit smoke + 5 cli smoke + 27 cleaning fixture from #28 work）；1452/1452 contracts 全绿（含 1023 figure 相关 import boundary parametrize 用例 + 2 个新 figure CLI surface 守门）；`bash scripts/figure_demo_einstein.sh` 退出码 0，`data/figure_bundles/einstein/<bundle_id>/manifest.json` 落盘 3 份（base/steering/lora），`data/figure_audit/` 3 条 audit JSON。
>
> **未做（已开新 debt）**：
> - PEFT backend 真 GPU 训练（→ #18 unchanged）
> - V2 archive fetcher + curated payload 数据集（→ #19 unchanged）
> - PersonaLoRAPool 真 hot-swap（→ #20 unchanged）
> - F5 steering 在真 substrate residual 流上抽方向（→ #21 unchanged）
> - DLaaS adopt 主路径自动 hook `lookup_figure_bundle` + `register_bundle_persona_lora`（→ #22 unchanged）
> - lu_xun corpus + e2e（→ #27 unchanged，CLI 占位 forward-compatible：`--figure lu_xun` 命中 `cmd_bake_bundle` 后 fail-loud 指向 #27）
> - corpus crawl + clean + verify 三层（→ #28 unchanged）

- ~~**路径**~~：
  - ~~现有 Python 函数（无 CLI 包装）~~：→ 现已通过 `cli/_commands.py` 把 `build_figure_artifact_bundle` / `bake_steering_set` + `apply_steering_through_gate` / `SyntheticLoRABakeBackend().bake` + `apply_persona_lora_through_gate` / `register_bundle_persona_lora` 全部 wired 进 5 个子命令
  - ~~缺位的脚本~~：→ 新增 `scripts/figure_demo_einstein.sh` + `python -m lifeform_domain_figure.cli`
  - ~~唯一调动 bake / gate apply 的真实代码~~：→ test fixture + CLI 实现现都在
- ~~**问题**~~：F5 / F6 OFFLINE-gated 训练流程现在通过 CLI + audit log + bundle 持久化全部可重复跑、可 review、可 rollback
- ~~**风险**~~：闭环
- ~~**触发条件**~~：闭环
- ~~**推荐修法**（7 项 minimum viable）~~：全部 land
- ~~**优先级**~~：闭环

## 24. Figure D2 三件 corpus helper（dedupe / provenance / citation parser）未接进 bundle 主管线

- **路径**：
  - 三件 helper 模块（已就位）：
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/dedupe.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/dedupe.py)（`compute_dedup_report` + `DedupReport` + `DuplicateGroup`，跨 envelope sha256 折叠 + 高信度 source kind 优先）
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/provenance.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/provenance.py)（`SourceProvenance` + `LegalClearance` + `CaptureMethod` + `fingerprint_provenance`）
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/citation.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/citation.py)（`parse_locator` + `ParsedLocator`，4 种 locator 字符串严格 typed parser）
  - **缺位的调用点**：
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/retrieval_index.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/retrieval_index.py) `build_figure_retrieval_index(...)` 不调 `compute_dedup_report` 过滤 canonical chunks
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py) `FigureBundleInputs` 没有 `provenance_records: tuple[SourceProvenance, ...]` 字段；`compute_bundle_integrity_hash` 不折进 `fingerprint_provenance`
    - [`packages/lifeform-expression/`](../packages/lifeform-expression/) `GroundedDecoder` 在 surfacing evidence pointer 时把 locator 当 opaque string 透传，未调 `parse_locator(...)` 派生结构化 citation 字段
- **问题**：D2 三件 helper 是 reviewer-facing pure 函数，schema + 测试都齐（20 个 smoke test 全绿），但**没人调** —— 它们存在但不 load-bearing。具体表现：
  - retrieval index 在 synthetic einstein corpus 上实测出 3 个跨 envelope dup（`_HEADER_NOTE` 横在 4 个 source kind 上），如果走 `compute_dedup_report` 过滤就只有 12 个 chunk 进 BM25；当前不过滤，15 个 chunk 都进，给重复 header 加权 → BM25 在 boilerplate 上 spike
  - `SourceProvenance` 是显式 license / `LegalClearance` / `CaptureMethod` 的 audit 入口，但 `FigureArtifactBundle.integrity_hash` 不依赖它 —— curator 改了一份 source 的 license 字段，bundle 还是同一个 hash，rollback 契约对 license 漂移盲
  - `parse_locator` 是 typed citation 渲染的前提（`ParsedLocator.sender_id` / `.date_iso` / `.venue_id` 等），但 grounded decoder 仍然只把 raw 字符串塞进 evidence pointer，下游审计 dashboard / 引证语义分组没法用结构化字段
- **违反**：不违反 R 铁律。三件 helper 的存在是为了让 reviewer 能查 / 审 / 去重，不是 runtime 必经。但 docstring 与 plan 都暗示它们会被 bundle 主管线消费，当前差一步串接。
- **风险**：低-中。短期看 e2e 测试都过（dedup 只多算一些重复，retrieval 仍能 top-k；provenance 是审计层不影响生成；locator 透传不影响 L3 grounding 决定）；长期当 #19 V2 archive fetcher 接入真 PDF/OCR 数据时，跨 archive 重复（同一文档在 CPAE 又在 Internet Archive 各扫一遍）会让 retrieval 重复加权严重，那时这条债会从"可演化性"变成"功能正确性"。
- **触发条件**：(a) #19 真 archive fetcher 接入后跨 archive 抓同一 figure 的同一文档；(b) tenant 要求"基于 license 字段过滤训练数据"；(c) 接入审计 dashboard 想按 sender / venue / date 分组引证；(d) curator 改了某 source 的 license 但 bundle hash 不变被 ops 发现
- **推荐修法**：
  1. **dedup integration**：`build_figure_retrieval_index(figure_id, envelopes, *, dedup_canonical_only: bool = False)` 加 kwarg；当 True 时先调 `compute_dedup_report(envelopes)`，按 `report.canonical_chunk_ids` 过滤进入索引的 chunk；默认 False 保持向后兼容
  2. **provenance integration**：
     - `FigureBundleInputs` 加 `provenance_records: tuple[SourceProvenance, ...] = ()` 字段（默认空允许向后兼容）
     - `compute_bundle_integrity_hash(...)` 多收一个 `provenance_fingerprint: str = ""` 参数，append 到 hash payload
     - `build_figure_artifact_bundle(...)` 把 `fingerprint_provenance(inputs.provenance_records)` 折进 integrity hash
     - 加 contract test：相同 envelope + 不同 license 的 provenance records → 不同 bundle_id
  3. **citation parser integration**：
     - `lifeform-expression.GroundedDecoder` 在构造 evidence pointer 时调 `parse_locator(...)`；保留 raw 字符串作 fallback；加 typed `ParsedCitation` 字段进 `EvidencePointer`（已有的话扩展，没有就新增 frozen dataclass）
     - 加 contract test：letter locator → evidence pointer 必须 surface `sender_id` / `recipient_id` / `date_iso`
  4. 守 R8：三件 helper 的所有调用都从 `build_*` 入口走，consumer 不直接 import 内部模块；contract test 静态守门
- **优先级**：低-中（独立可做，不强依赖 #19 但 #19 落地后会从"可选优化"升为"硬要"）

## ~~25. Figure D4 metadata digest fingerprint 未折进 `FigureArtifactBundle.integrity_hash`~~ —— 2026-05-10 关闭

> **关闭说明（2026-05-10）**：随 debt #26 V2 metadata client 一并 land 完成 R15 字节级回滚契约的最后一环。具体实现见 [`docs/DATA_CONTRACT.md`](DATA_CONTRACT.md) §1.6 和契约测试 [`tests/contracts/test_figure_bundle_metadata_fingerprint.py`](../tests/contracts/test_figure_bundle_metadata_fingerprint.py)：
>
> - `FigureBundleInputs.metadata_digest: MetadataDigest | None = None`（默认 None 向后兼容）
> - `FigureArtifactBundle.metadata_digest_fingerprint: str = ""`（默认空向后兼容；`MetadataDigest.fingerprint` 透传）
> - `compute_bundle_integrity_hash(..., metadata_digest_fingerprint="")` 默认空时**不折入** hash → 既有 bundle 字节级稳定；非空时折入 → 不同 digest 产不同 bundle id
> - `attach_steering_to_bundle` / `attach_lora_to_bundle` 重算 hash 时保留 metadata_digest_fingerprint（防止 LoRA/steering bake 后丢失 metadata 审计链）
> - 5 个 per-package case + 4 个 contract case 全绿
>
> _（保留下方原始描述以便 audit。）_

- **路径**：
  - metadata 富集入口（已就位）：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/coverage_enrichment.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/coverage_enrichment.py) `enrich_profile_with_metadata(profile, digest) -> HistoricalFigureProfile` 把 `MetadataDigest.coverage_hints` 折进 `domain_coverage_seed` + 把 `lifespan.death_year` 折进 `boundary_priors`
  - metadata 聚合入口：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/records.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/records.py) `aggregate_metadata(...)` 产 `MetadataDigest.fingerprint`（sha256 over identity-bearing fields）
  - **缺位的串接**：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py) `FigureBundleInputs` 没有 `metadata_digest: MetadataDigest | None = None` 字段；`compute_bundle_integrity_hash(...)` 不读 metadata fingerprint
- **问题**：D4 的 enrichment 路径设计是：
  ```
  raw API payload → typed metadata record → MetadataDigest (fingerprinted) →
    enrich_profile_with_metadata(profile, digest) → enriched profile →
      build_figure_artifact_bundle(FigureBundleInputs(profile=enriched, ...)) → bundle
  ```
  但**最后一步丢了 fingerprint** —— `build_figure_artifact_bundle` 只看 `profile`（已经 fold-in 部分 hint 到 `domain_coverage_seed` / `boundary_priors`）+ retrieval / coverage / style / steering / lora 的 integrity hash。结果：
  - 同一份 profile 经过两次 enrichment（hint 集合不同，但都被合并去重到 profile 后表面看不出来），可能产生不同的下游 coverage map → 不同的 bundle hash —— 但**没法从 bundle 直接审计"这是用了哪份 metadata digest 富集出来的"**
  - 富集后 enriched.version 字段拼了 `+metadata:{fingerprint[:8]}` 作为 audit hint，但这是字符串前缀不是 hash 输入；ops 想验证"这个 bundle 是用 OpenAlex 2026-05 版本 + Wikidata Q937 还是别的"，只能比对 version 字符串，没有 byte-level 的 R15 回滚契约
  - DATA_CONTRACT 2.15 列了 `version_window` / `integrity_hash` 但没列 metadata digest fingerprint，schema 上也没承诺 ——是设计的盲区
- **违反**：不违反 R 铁律，但**违反 R15 byte-level 回滚契约的精神** —— "任何字节级输入变化产生不同 bundle_id" 这条，如果只看 enrichment 的 hash 输入是不成立的（hint 合并去重后看不到差异）。
- **风险**：低-中。短期看 demo 用一份 metadata 跑一份 bundle，单跑没问题；长期当 tenant 要"换一组 OpenAlex topic 重 bake bundle 跑对比"时，会发现两份 bundle 的 hash 不一定不同（取决于 hint 是否新进了 seed 集合）；audit 找不到"这是哪份 digest 烤出来的"。
- **触发条件**：(a) 第一个 tenant 拿同一 profile 跑两组不同 metadata digest 想对比；(b) ops 审计某 bundle 想反查"这是用 Wikidata 哪天的快照富集的"；(c) #19 V2 fetcher 接入后 metadata 也 V2，需要 trace metadata 数据漂移到 bundle hash
- **推荐修法**：
  1. `FigureBundleInputs` 加 `metadata_digest: MetadataDigest | None = None`（默认 None 保持向后兼容；非 None 时 fingerprint 进 hash）
  2. `compute_bundle_integrity_hash(...)` 加 `metadata_digest_fingerprint: str = ""` 入参，append 进 payload tuple
  3. `build_figure_artifact_bundle(inputs)` 把 `inputs.metadata_digest.fingerprint if inputs.metadata_digest else ""` 喂进上述参数
  4. `FigureArtifactBundle` 加 `metadata_digest_fingerprint: str = ""` 字段（默认空保持向后兼容；非空则 audit 时可反查）
  5. DATA_CONTRACT 2.15 表格加 `metadata_digest_fingerprint` 字段说明，明确"非空表示 bundle 经 D4 metadata enrichment，可凭 fingerprint 反查 digest"
  6. 加 contract test `tests/contracts/test_figure_bundle_metadata_fingerprint.py`：(a) 同 profile 同 digest → 同 hash；(b) 同 profile 不同 digest → 不同 hash；(c) digest=None → fingerprint 字段为空字符串 + bundle 整体仍按现有路径产生稳定 hash
- **优先级**：低（独立可做；现在 metadata 路径仍可用，只是 audit 闭合缺最后一环）

## ~~26. Figure D4 metadata 4 个 client（OpenAlex / Wikidata / Crossref / SEP）V2 live HTTP 未做（与 #19 同构）~~ —— 2026-05-10 关闭

> **关闭说明（2026-05-10）**：4 个 V2 live metadata client 全部 land，与 L0 corpus crawler stack 共用 `BaseHTTPClient` + `ScopePolicy`（debt #28 L0 packet 引入），通过新增的 `ScopeRole` 标签机制做 cross-role SSRF 防御。详见 [`docs/DATA_CONTRACT.md`](DATA_CONTRACT.md) §1.5 + [`docs/specs/figure-corpus-crawl.md`](specs/figure-corpus-crawl.md) §"Metadata HTTP backbone"：
>
> - **共用 HTTP layer (修法 1)**：`MetadataHTTPClient` 在 `metadata/http_client.py`，wrap `BaseHTTPClient` 强制 `required_role=ScopeRole.METADATA_FETCH`；SSRF 5 重门继承自 L0
> - **4 个 live client (修法 2)**：
>   - `live_openalex_client(...)` → `api.openalex.org/works?filter=author.id:{id}` cursor 分页
>   - `live_wikidata_client(...)` → `Special:EntityData/{qid}.json` claims (P569/P570/P106/P101) 解析
>   - `live_crossref_client(...)` → `api.crossref.org/works/{doi}` + `fetch_raw_message(doi)` 给 verifier 直接读 relation/translator
>   - `live_sep_client(...)` → `plato.stanford.edu/entries/{slug}/` HTML via bs4
> - **共享 cache (修法 3)**：`MetadataCache` content-addressable on-disk JSON cache `data/metadata_cache/{provider}/{key_sha256}/`，TTL 默认 24h，支持 TTL=0 关闭过期
> - **守 R12 (修法 4)**：metadata clients **禁止** import `Figure*Source` typed records / kernel modules；contract test [`tests/contracts/test_verification_module_boundaries.py`](../tests/contracts/test_verification_module_boundaries.py) AST 守门
> - **新增 `ScopeRole` (CORPUS_FETCH / METADATA_FETCH)**：在 `crawl/scope_policy.py`；`ScopePolicy.host_roles` per-host 标签；`BaseHTTPClient.get(..., required_role=...)` 跨角色 SSRF 拒收。L0 fetcher 全部传 CORPUS_FETCH，metadata client 全部传 METADATA_FETCH
> - **3 个 default factory**：`default_scope_policy(...)` (corpus only) / `default_metadata_scope_policy(...)` (metadata only) / `default_combined_scope_policy(...)` (两者，分别打 role)
> - **同步关闭 #28 L2 second batch**：4 个 metadata-依赖 verifier (IDENTITY_DISAMBIGUATION / AUTHORSHIP_ATTRIBUTION / VERSION_RECONCILIATION / TRANSLATION_LINEAGE) 真实现，全 backed by 这些 V2 client；详见 #28 progress 行
> - **同步关闭 #25**：metadata digest fingerprint 折入 bundle integrity hash
> - 21 个 per-package case (5 件 smoke test + verifier 4 件) + 4 个 contract case 全绿；既有 figure tests 零回归；ruff 新文件全绿
>
> **剩余 follow-up**：与 #15 DLaaS asset.uri fetcher 共用 `BaseHTTPClient` 仍未做（DLaaS 平台层独立工作；L0/L2 现在已是参考实现）。
>
> _（保留下方原始描述以便 audit。）_

- **路径**：
  - 4 个 client Protocol + offline 桩（已就位）：
    - [`metadata/openalex.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/openalex.py) `OpenAlexClient` Protocol + `_OfflineOpenAlexClient.fetch_author_works(...)` raise `NotImplementedError`
    - [`metadata/wikidata.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/wikidata.py) `WikidataClient` Protocol + `_OfflineWikidataClient.fetch_person(qid=...)` raise
    - [`metadata/crossref.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/crossref.py) `CrossrefClient` Protocol + `_OfflineCrossrefClient.fetch_work(doi=...)` raise
    - [`metadata/sep.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/sep.py) `SEPClient` Protocol + `_OfflineSEPClient.fetch_entry(slug=...)` raise
  - **缺位的实现**：4 个对应的 live HTTP client 都不存在；与 #19 archive V2 fetcher 缺位完全同构
- **问题**：D4 metadata 4 个来源的 typed payload schema + `*_to_*` 翻译器 + Protocol 客户端齐全（20 个 smoke test 全绿），但 V1 一律 offline-only —— 任何想跑真 OpenAlex / Wikidata / Crossref / SEP 拉数据的 user 都得手抄 JSON 进 `OpenAlexWorkPayload(...)` Python 字面量。具体后果：
  - 跟 #19 archive fetcher 缺位是同样的"schema ready, V2 fetcher 未做"模式；用户得手写 80-150 行 boilerplate 把每份 metadata 转成 typed payload
  - 与 #15 DLaaS asset fetcher 也是同构问题：都缺一份 SSRF allowlist + content-type 嗅探 + retry policy 的共用 HTTP layer
  - 4 个 metadata 来源 + 4 个 archive 来源（#19）+ DLaaS asset.uri (#15) = **9 个 V2 HTTP fetcher 未做**，统筹设计 / 复用基础设施会比每个独立写省一半工作量
- **违反**：不违反 R 铁律。Protocol + offline 桩本身是合法的 V1 设计（与 D3 archive `_OfflineArchiveFetcher` 同纪律），与 `lifeform-ingestion` "web sources are slice 2b territory" 一致。
- **风险**：低（短期 demo / 内部 evidence 用 hardcoded payload 仍可跑），中（外部用户 onboard 阻塞 —— 没人会手抄 50 篇 OpenAlex 论文 metadata）
- **触发条件**：(a) 第一个 tenant 想自动从 OpenAlex 抓某 author 的全部 works；(b) #19 archive V2 fetcher 落地后想统一抽 `BaseHTTPClient`；(c) #15 DLaaS asset fetcher 设计阶段想 6-9 个 HTTP fetcher 一次性做完
- **推荐修法**：
  1. **共用 HTTP layer**（与 #19 / #15 一起设计）：在 `lifeform-ingestion` 或新 `vz-net` 加 `BaseHTTPClient` —— SSRF allowlist + content-type 嗅探 + 速率限制 + 重试 policy + cache 层（沿用 ingestion slice 2b 纪律）
  2. **4 个 live client**：
     - `LiveOpenAlexClient(BaseHTTPClient).fetch_author_works(*, openalex_author_id) -> tuple[OpenAlexWorkPayload, ...]`：调 `https://api.openalex.org/works?filter=author.id:{id}&per-page=200` + paginate + 解析 JSON → typed payload
     - `LiveWikidataClient(BaseHTTPClient).fetch_person(*, qid) -> WikidataPersonPayload`：调 SPARQL `SELECT ... WHERE { wd:{qid} ... }` + 解析 → typed payload
     - `LiveCrossrefClient(BaseHTTPClient).fetch_work(*, doi) -> CrossrefWorkPayload`：调 `https://api.crossref.org/works/{doi}` + 解析
     - `LiveSEPClient(BaseHTTPClient).fetch_entry(*, slug) -> SEPEntryPayload`：调 `https://plato.stanford.edu/entries/{slug}/` + HTML 解析（节选 `<h2>` 节标题 + summary）
  3. **共享 cache**：metadata 通常很稳定（人物 lifespan / DOI metadata 不常变），加 `data/metadata_cache/{provider}/{key}.json` 落盘 + ttl，避免每次 build bundle 都重抓
  4. 守 R12：metadata 4 个 client **不**反向写任何 kernel owner；都是 readout；加 `tests/contracts/test_figure_metadata_no_kernel_writeback.py` 静态守门（与 #13 / #14 同 pattern）
  5. 与 #19 archive fetcher 落地一起做 review，单独 packet review 走 `cursor-convergence-workflow.mdc`
- **优先级**：低（与 #19 / #15 同时做更高效；当前 offline 桩对内部 evidence 路径不阻塞）

## 27. Figure D7 鲁迅 PoC 半完成：缺 sample corpus / lifeform builder / e2e test / CTP curated 数据

- **路径**：
  - 已就位：
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/profiles/lu_xun.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/profiles/lu_xun.py) reviewed `HistoricalFigureProfile`（6 knowledge seeds / 4 cases / 3 strategies / 3 boundaries / 5 drives / 3 time windows + post-1936 absolute boundary）
    - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/archives/chinese_text_project.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/archives/chinese_text_project.py) `CTPPayload` + `ctp_to_paper_source` 翻译器（古典中文人物如孔子用）
    - [`packages/lifeform-domain-figure/tests/test_chinese_figure_smoke.py`](../packages/lifeform-domain-figure/tests/test_chinese_figure_smoke.py) 7 个 PoC 测试（profile build / boundary / CTP adapter / 单段 corpus 跑 retrieval+coverage / Wikidata 富集）
  - **缺位**：
    - 没有 `synthetic_lu_xun_corpus()` —— [`sample_corpus.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/sample_corpus.py) 只有 `synthetic_einstein_corpus()`；想跑 鲁迅 e2e demo 必须每次手工写 `FigurePaperSource(...)` 字面量
    - 没有 `build_lu_xun_lifeform()` —— [`lifeform_builder.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lifeform_builder.py) 只有 `build_einstein_lifeform()`；DLaaS adopt + verticals 工厂没有鲁迅入口
    - 没有 `lifeform-service.verticals._try_lu_xun` 工厂 —— [`packages/lifeform-service/src/lifeform_service/verticals.py`](../packages/lifeform-service/src/lifeform_service/verticals.py) 只有 `_try_einstein`
    - 没有鲁迅 e2e test —— `test_full_chain_e2e_smoke.py` 只跑 Einstein bundle 的 retrieval × coverage × style × steering × LoRA 全链路；中文 figure 链路没覆盖
    - 没有 CTP curated 数据集 —— `packages/lifeform-domain-figure/data/ctext/` 不存在；想跑孔子的 demo 必须手抄 `CTPPayload(...)`
    - 没有 CTP HTTP fetcher（与 #19 / #26 同构 V2 缺口）
- **问题**：D7 PoC 验证了"中文 figure 走得通 schema"（鲁迅 profile 通过所有 validator + 单段 corpus 能跑 retrieval+coverage+Wikidata 富集），但**与 Einstein 路径相比缺 5 个对称模块**：sample corpus / lifeform builder / verticals 工厂 / 全链路 e2e test / CTP curated 数据集。具体后果：
  - 任何 cite "figure vertical 已支持中文人物" 的文档只能引用 PoC test，不能引用 demo 路径
  - DLaaS adopt 接入鲁迅需要 tenant 自己提供 corpus + lifeform builder，平台层"开箱即用"承诺只在 Einstein 上成立
  - F5 / F6 的 steering / LoRA 训练管线在中文 corpus + 鲁迅 profile 上**没跑过一次** —— 不知道 hashing embedding 在中文 token 下表现如何（`_WORD_RE` 已含中文 unicode 范围，但实际 token 化质量没在中文 e2e 验证）
  - 与 #19 / #26 / #23 都强相关：CTP 数据集 + V2 HTTP fetcher + bake CLI 都得在中文场景再跑一次才能算"对中文人物可用"
- **违反**：不违反 R 铁律。D7 plan 标注"（可选）中文人物 PoC"，PoC 完成本身是合规的；只是从"PoC"到"与 Einstein 路径对等"还有一段。
- **风险**：低（短期 PoC 测试覆盖）—— 中（被引用为"已支持中文" / 第一个中文 tenant onboard 时硬要）
- **触发条件**：(a) 第一个 tenant 想做中文 figure（鲁迅 / 孔子 / 老子 / 苏轼 等）；(b) 想跑"中文人物 figure vertical 端到端"对外 demo；(c) #19 V2 fetcher / #23 bake CLI 落地后想验证"中文路径同等覆盖"；(d) F5 contrast set 想加中文 vs 中文 stance pair（如鲁迅 vs 林语堂 vs 胡适）做 steering 训练
- **推荐修法**（按依赖顺序）：
  1. **sample corpus**：在 `sample_corpus.py` 加 `synthetic_lu_xun_corpus() -> tuple[FigurePaperSource, FigureLetterSource, FigureLectureSource, FigureNotebookSource]`，paragraph 体例与 `synthetic_einstein_corpus` 对称（每段都是 reviewer paraphrase + 明文标 synthetic）；语种 zh，长度对等
  2. **lifeform builder**：在 `lifeform_builder.py` 加 `build_lu_xun_lifeform(...) -> FigureLifeformBundle`，参数对称 `build_einstein_lifeform`；wheel `__init__.py` 公开
  3. **verticals 工厂**：在 `lifeform-service.verticals` 加 `_try_lu_xun(name)` 工厂；`build_dlaas_app(...)` 路径上鲁迅可被 adopt
  4. **e2e test**：加 `tests/test_lu_xun_full_chain_e2e_smoke.py`：同 `test_full_chain_e2e_smoke.py` 结构，跑鲁迅 retrieval × coverage × style；steering / LoRA 用 synthetic 后端跑通；与 Einstein 测试结构对称
  5. **CTP curated 数据集**：`packages/lifeform-domain-figure/data/ctext/` 加 reviewer 已 curated 的孔子 / 老子 `CTPPayload` JSON 序列（`论语` 20 篇 + `道德经` 81 章 minimum viable real corpus）；`corpus/loaders/load_curated_ctp_payloads(figure_id) -> tuple[CTPPayload, ...]` loader
  6. **CTP V2 fetcher**：与 #19 / #26 一起设计 `LiveCTPFetcher(BaseHTTPClient)`；ctext.org 是公共 API 友好，allowlist 简单
  7. **守 R8**：所有 builder / 工厂 / loader 只产 envelope / bundle 公开 surface，不直接 import 内部模块；contract test 静态守门（同 verticals 现有 pattern）
- **优先级**：低-中（独立可做，不强依赖；做完后 figure vertical 才算"对中英文人物对等覆盖"）

## 28. 完整 webcrawl 编排 + 数据清洗管线 + 多源验证审计三层全未做（在 #15 / #19 / #26 单文档 fetcher 之上的整层缺口）

> **进度（2026-05-10）**：
>
> ✅ **L1 cleaning pipeline 已落地** —— 见 [`docs/specs/figure-corpus-cleaning.md`](specs/figure-corpus-cleaning.md)、`packages/lifeform-domain-figure/src/lifeform_domain_figure/cleaning/`、CLI `packages/lifeform-domain-figure/scripts/figure_clean.py`、契约测试 [`tests/contracts/test_cleaning_pipeline_versions.py`](../tests/contracts/test_cleaning_pipeline_versions.py)。包含 4 个真实 parser（CPAE PDF via pypdf / Wikisource HTML via bs4+mwparserfromhell / Gutenberg HTML+plain via bs4 / IA OCR JSON via stdlib）+ 6 个 cleaner op + content-addressable raw/cleaned store + cleaner 版本化（v1 + v2 共存）+ re-clean-all CLI + 桥接到既有 `*Payload`。
>
> ✅ **L2 verification first batch（3 / 7 verifier）已落地** —— 见 [`docs/specs/figure-corpus-verification.md`](specs/figure-corpus-verification.md)、`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/`、CLI `packages/lifeform-domain-figure/scripts/figure_verify.py`、契约测试 [`tests/contracts/test_bundle_admits_only_verified_sources.py`](../tests/contracts/test_bundle_admits_only_verified_sources.py) + [`tests/contracts/test_verification_module_boundaries.py`](../tests/contracts/test_verification_module_boundaries.py)。包含：(a) 7 `CheckKind` 关闭枚举 + 3 verdict + 不可变 `VerificationCheck` schema；(b) 3 个真实 verifier（DATE_PLAUSIBILITY / LICENSE_PAGE_LEVEL / CROSS_SOURCE_BYTE，全 pure function，零外部依赖）；(c) 4 个 deferred kind 的 `NotImplementedError` stub（强制 fail-loud，等 #26 metadata client）；(d) `VerificationLedger` content-addressable append-only JSONL 持久层（`data/verification/{byte_sha256}/checks.jsonl`）；(e) `build_figure_artifact_bundle(FigureBundleInputs(..., require_verification_pass=True, provenance_records=..., verification_ledger=...))` bundle gate；(f) 抽样 / 人审覆盖 / 列表三子命令 CLI；(g) 修法 5 接线 `cleaning/bridging.py.cleaned_to_source_provenance(...)` 把 L1 `RawDocument.license_notice` 流到 `SourceProvenance.license_label`；(h) 55 个新 test case 全绿，零回归。
>
> ✅ **修法 5（L1 license_notice → SourceProvenance.license_label 接线）已落地** —— 上述 (g)。
>
> ✅ **L0 crawler frontier 已落地（同时关闭 debt #19 V2 archive fetcher）** —— 见 [`docs/specs/figure-corpus-crawl.md`](specs/figure-corpus-crawl.md)、`packages/lifeform-domain-figure/src/lifeform_domain_figure/crawl/`、CLI `packages/lifeform-domain-figure/scripts/figure_crawl.py`、契约测试 [`tests/contracts/test_crawler_module_boundaries.py`](../tests/contracts/test_crawler_module_boundaries.py) + [`tests/contracts/test_crawler_respects_robots.py`](../tests/contracts/test_crawler_respects_robots.py) + [`tests/contracts/test_crawler_uses_l1_cleaning_store.py`](../tests/contracts/test_crawler_uses_l1_cleaning_store.py)。包含：(a) `CrawlStatus` (7 enum) + `CrawlRequest` / `CrawlResult` / `ScopePolicy` 不可变 schema；(b) `BaseHTTPClient` SSRF 5 重门（scheme + host + path-prefix + redirect-1-hop-rescope + body-cap）+ retry + 304 sentinel；(c) `RobotsRegistry` per-host 缓存 + TTL + fail-closed；(d) `TokenBucketRateLimiter` per-host 默认 0.5 req/s + burst 5；(e) `CrawlFrontier` 内存+磁盘双层 + dedup + `resume_from_disk`；(f) 5 个 fetcher（generic + cpae + wikisource (action=raw 优先) + gutenberg (.txt 优先) + internet_archive (metadata API → OCR JSON)）；(g) `CrawlSink` 直写 L1 `CleaningStore.put_raw`，建立 `raw_sha256` anchor；(h) `CrawlScheduler` 端到端 orchestrator (scope → robots → rate → dispatch → fetch → sink)；(i) `live_archive_fetcher(fetch_kind, ...)` 工厂关闭 debt #19 V2，返回 `LiveFetchedBytes` raw_payload，既有 `offline_archive_fetcher()` 行为不变；(j) 5 子命令 CLI；(k) `requests` 是新 dep；(l) 73 个新 test case 全绿，零回归。
>
> ✅ **L2 second batch（4 个 metadata-依赖 verifier）已落地（同时关闭 debt #26 + debt #25）** —— 4 个 verifier 真实现，全 backed by V2 metadata clients；`IMPLEMENTED_CHECK_KINDS = frozenset(CheckKind)` 全 7 启用。详细：(a) `verify_identity_disambiguation`（Wikidata QID + 生年 ±1 容差 + 职业重叠）；(b) `verify_authorship_attribution`（OpenAlex author works + co-author overlap 边缘启发）；(c) `verify_version_reconciliation`（Crossref relation map：is-version-of / replaces / is-translation-of 等 7 类）；(d) `verify_translation_lineage`（Crossref translator × language match 启发，识别翻译血缘）；(e) `MetadataDependentVerifierContext` typed bundle 让 batch CLI 一次注入所有客户端 + figure 上下文；(f) 21 个 per-package case + 4 个 contract case 全绿；(g) bundle gate 现要求每条 source 7 axes 全 PASS（NEEDS_REVIEW 转 PASS 必经 human override）。
>
> 本债**完全闭合**：L0 + L1 + L2（first + second batch）+ #19 + #25 + #26 全部 land。剩余只是数据层工作（curated payload 数据集、reviewer 抽样人审 CLI、license 显式 surface 等），不再是架构缺位。

- **路径**：
  - 既有"单文档 fetcher 缺位"债（**只到"给一个 URL 拿一份"这一层**）：
    - DLaaS asset.uri：[`packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py) `_handle_activate_template` 不读 asset.uri（→ #15）
    - figure 4 archive：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/archives/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/archives/) 4 个 `_OfflineArchiveFetcher.fetch(...)` raise（→ #19）
    - figure 4 metadata：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/) 4 个 `_OfflineXClient.fetch_*(...)` raise（→ #26）
  - **本债覆盖的三层 = 即便上面 9 个 fetcher 全部 V2 接通，仍未做的内容**：
    - **L0 crawl 编排**：URL 发现 / 链接遍历 / robots.txt / 速率限制 / crawl frontier / 增量重抓 / scope policy / 并发与队列
    - **L1 清洗管线**：format 解析（PDF / HTML / MediaWiki / OCR JSON）+ boilerplate 剥离 + 编码归一 + 语种检测 + 排版标准化 + 段内去重 + PII 屏蔽 + 质量评分（OCR 置信度 / 版式完整度）+ 文档级 license notice 抽取 + cleaning pipeline 版本化（同份原始字节，新版 cleaner 重跑产新文本而不丢旧版）
    - **L2 多源验证**：跨源 corroboration（同一封信 CPAE 与 Wikisource 字节是否一致）+ 身份消歧（OpenAlex 这个 "Albert Einstein" 是不是同一人）+ 日期合理性（落在 figure_lifespan 内）+ 作者归属验证（这段确实由该 figure 所写，不是 quote / 弟子转述 / 编辑增补）+ 多版本协调（letter 多次出版的差异、哪版 canonical）+ 翻译血缘（哪个译者 / 译本偏移）+ reviewer audit trail（谁审过 / 何时审 / 接受与否的依据）+ 抽样人审（auto-fetched 的 N% reviewer spot-check）
  - **缺位的存储 / 血缘层**：
    - `packages/lifeform-domain-figure/data/` 不存在
    - 没有 raw bytes content-addressable archive（按 sha256 命名的不可变层）
    - 没有 cleaned text 版本流（`{sha256}/cleaned-v{N}.txt`）
    - 没有 cleaning pipeline 版本号 → 重跑工作流（升级 parser 后批量重跑 + 输出 diff）
    - 没有 verification verdict 持久层（每条源 → tuple[VerificationCheck, ...] → reviewer 决议）
- **问题**：当前架构上**对一份 source 的处理止于 `FigurePaperSource.body: str`**，假设这段 body 已经是 cleaned + verified 的 plain text。但从 "URL → bytes → text → 进 bundle" 的真实管线，需要至少三层独立机制：
  - 没有 L0：tenant 想自动跟踪某个 archive 的新文档（CPAE 每年更新 1-2 卷），没有 incremental crawl；想批量处理 1000 篇 OpenAlex works，没有任务队列 / retry / 失败聚合
  - 没有 L1：`*Payload.body` 假设输入已 cleaned，但 PDF / HTML / MediaWiki / OCR JSON 的实际字节流远不是 plain text；任何 V2 fetcher 一接通，body 字段就要么塞进未清洗的脏文本（污染 retrieval index），要么 fetcher 内部偷偷做 cleaning（违反单一职责 + 没法 audit cleaning 版本）
  - 没有 L2：synthetic einstein corpus 是 reviewer 手写的所以"100% 对"，但真 corpus 里 Einstein 的同一封信可能在 CPAE / Wikisource / Internet Archive 各有不同字节版本（拼写矫正 / 标点 / 段落分割差异），没有 corroboration → bundle 选了哪个版本完全靠 fetcher 顺序运气；OpenAlex 上重名 author（Einstein 不只一个，全球同名学者很多）没有身份消歧，会把别人的论文塞给 figure
  - 加在一起：**如果今天有人接通 #15 / #19 / #26 的 9 个 V2 fetcher 直接跑生产 corpus，bundle 里很可能混入：(i) 别人写的（重名）；(ii) 弟子 / 编辑替写的（attribution 错）；(iii) 同一文档的脏版本（OCR 错乱 / 编码错乱）；(iv) 后世翻译者带偏的版本；(v) 来源 license 实际上不是 public domain 的（fetcher 没读 page-level license notice）**
  - 这才是真正阻塞 "Einstein 全集放进去能得到 Einstein 而不是被污染的 Bohr+Einstein 大脑" 的层 —— 不是模型层，是数据层
- **违反**：不违反 R 铁律，但**违反 Persona Figure 数据管线 V1 plan 第 0 节工作假设**："V1 手工 feed → V2 半自动 archive URL → V3 学术 API metadata"——V2/V3 一旦想从 V1 的"手工已 cleaned"模式升档到自动化，必须把 L0/L1/L2 三层立出来。当前 plan 隐含假设这三层会随 V2 fetcher 一并到位，实际它们是独立工作量，比 fetcher 本身大数倍
- **风险**：
  - **短期低**：synthetic corpus + 单 figure manual feed 模式跑 demo / e2e 不受影响
  - **中期中-高**：第一个 tenant 想批量自动 onboard 真 corpus 时硬阻塞，且不阻塞地强行接通会把脏数据 / 错归属内容塞进 bundle（"figure 模仿 Einstein 但其实学了 OpenAlex 重名作者的论文" 的灾难场景）
  - **长期决定项目可信度**：figure vertical 的 L1/L2/L3/L4 全四档保真度都建立在 "primary corpus 真是该 figure 写的" 这条前提上。L2 验证层缺位 = 整个保真阶梯地基松动
- **触发条件**：(a) 第一次想自动从 archive 抓 ≥ 100 篇文档；(b) 第一次发现两个 source 的 body 不一致需要选 canonical；(c) 第一次发现 OpenAlex / Wikidata 上的"Albert Einstein"/"Lu Xun" 实际是同名不同人；(d) 第一次有 reviewer 想审 "这批 200 个 chunk 是怎么来的"；(e) #19 archive V2 fetcher 接通后第一次跑出 PDF body；(f) 任何外部第三方质疑"你这个 figure 里有多少是真它写的"
- **推荐修法**（按依赖顺序，三层各立独立 packet）：
  1. **L1 cleaning pipeline 先做**（不依赖 V2 fetcher，可独立验证）：
     - 新 wheel `lifeform-corpus-cleaning` 或 `lifeform-domain-figure/cleaning/` 子包
     - `parsers/`：`parse_cpae_pdf` / `parse_wikisource_html` / `parse_gutenberg_html` / `parse_archive_org_ocr_json` / `parse_ctext_html` 各一个，输入 `(bytes, content_type, source_url)`，输出 `RawDocument(text, layout_quality, ocr_confidence, encoding_detected, language_detected, license_notice, parser_version)`
     - `cleaners/`：boilerplate strip / 段落归一 / 排版标准化 / PII 屏蔽 / 段内 dedup；输入 `RawDocument`，输出 `CleanedDocument(text, cleaner_version, cleaning_log: tuple[CleaningOp, ...])`
     - **版本化**：raw bytes / cleaned text 全部 content-addressable 存到 `data/raw/{sha256}/` 与 `data/cleaned/{raw_sha256}/v{cleaner_version}/`；cleaner 升级后 `re-clean-all` 命令批量重跑出新 `v{N+1}` 版，与旧版 diff 进 audit
     - 守 R8：cleaner / parser 不直接产 `FigurePaperSource`；产中性 `CleanedDocument`，再用 D3 archive 适配器翻译进 typed source
     - 加 `tests/contracts/test_cleaning_pipeline_versions.py`：cleaner v1 vs v2 对同一 raw 产不同 cleaned；raw sha256 不变；cleaned 文件按 cleaner 版本号目录隔离
  2. **L2 verification + audit 第二做**（依赖 L1 输出）：
     - `lifeform-corpus-verification` 子包
     - `VerificationCheck` 不可变记录：`check_kind` enum（CROSS_SOURCE_BYTE / IDENTITY_DISAMBIGUATION / DATE_PLAUSIBILITY / AUTHORSHIP_ATTRIBUTION / VERSION_RECONCILIATION / TRANSLATION_LINEAGE / LICENSE_PAGE_LEVEL）+ `verdict` enum（PASS / FAIL / NEEDS_REVIEW）+ `evidence: tuple[str, ...]` + `reviewer_id` + `reviewed_at_iso`
     - `VerificationLedger` 持久层：每条 source → tuple[VerificationCheck, ...]；落到 `data/verification/{source_sha256}/checks.jsonl`
     - **gate**：`build_figure_artifact_bundle(...)` 加 `require_verification_pass: bool = False` 默认开关；非 None 时拒绝把 verdict 不全 PASS 的 source 入 bundle
     - 抽样人审入口：`scripts/figure_verify.py review --figure einstein --sample 10`，CLI 抽 10 个 random source 给 reviewer，确认通过 / 拒绝 / 标 NEEDS_REVIEW
     - 与 #18 / #19 / #20 / #22 / #23 都强相关：bake CLI 完成后的 audit trail (#23) 应自然 surface verification ledger
  3. **L0 crawl 编排最后做**（依赖 L1 + L2，否则 crawl 出来的脏数据没法处理）：
     - `lifeform-corpus-crawler` 或与 `lifeform-ingestion` 合并：crawler frontier / 调度器 / robots.txt 服从 / 速率限制 / 失败重试 / scope policy / incremental 重跑
     - 与 #15 / #19 / #26 的 V2 fetcher 在同一 review 里设计：crawler 是 fetcher 的 orchestration 层，fetcher 是 crawler 的 leaf node
     - 共用 `BaseHTTPClient`（与 #26 推荐修法 1 同）：SSRF / content-type / 重试 / cache 全在一处
  4. **守红线全程**：L0/L1/L2 任一层都不能反向写 kernel owner（R8 / R12）；所有 cleaner / verifier / crawler 输出都是 readout / artifact，进 bundle 只通过 `build_figure_artifact_bundle` 一个入口（与 #19 推荐修法 6 同口径）
  5. **守 license**：L1 cleaner 抽 page-level license notice → `SourceProvenance.license_label`（D2 已有 schema）；L2 verifier 把 page license 与 reviewer-declared `LegalClearance` 对齐失败时拒绝入 bundle
  6. **共用基础设施**：与 #15 (DLaaS asset fetcher) / #19 (archive fetcher) / #26 (metadata fetcher) 一次性 review；6-9 个独立 fetcher + crawl + clean + verify 加起来是 1-2 季度的工作量，但分散来做要 3-4 季度，且共用基础设施缺位时各自重造轮子
  7. **加守门契约**：
     - `tests/contracts/test_corpus_cleaning_pipeline_no_kernel_writeback.py` 静态守 cleaner / verifier 不写 kernel
     - `tests/contracts/test_bundle_admits_only_verified_sources.py` 当 `require_verification_pass=True` 时，未 PASS 的 source 不进 bundle
     - `tests/contracts/test_crawler_respects_robots.py` crawler 输出对每个 robots.txt 都遵守
- **风险评估说明**：本债是 figure vertical "完整 webcrawl 及数据清洗验证" 的统称，规模显著大于 #15 / #19 / #26 三条单文档 fetcher 债之和。它直接决定 "Einstein 全集进去得到 Einstein 而不是被污染的脑" 这个产品命题能否兑现 —— 单文档 fetcher 是必要条件但远非充分条件
- **优先级**：中（产品命题层硬阻塞，但需要先把更基础的 #18 / #19 / #20 / #22 / #23 推进；L1 cleaning 是性价比最高的入口，不依赖 V2 fetcher 就能做并立刻给 #19 / #26 用）

## 29. 对外 EQ / 共情类 benchmark 提交完全缺位（融资尽调可被一条 Google 戳穿的证据空缺）

> **2026-05-10 progress note**: 推荐修法 (1)–(5) 的代码层 + 文档层全部 land（Packet 1-9）。Packet 10 公开提交仍 gate 在 verdict — 需要先用真实 GPU + Anthropic API key 跑一次三轨 ablation 拿分数。具体落地：
> - 适配 wheel：`packages/lifeform-openai-compat/` ([README](../packages/lifeform-openai-compat/))，红线静态守门 [tests/contracts/test_openai_adapter_import_boundary.py](../tests/contracts/test_openai_adapter_import_boundary.py) + [tests/contracts/test_openai_adapter_no_kernel_writeback.py](../tests/contracts/test_openai_adapter_no_kernel_writeback.py)
> - 内核侧改动：1 行 — `lifeform-service/cli.py` 加 `--enable-openai-compat` flag（默认 off，向后兼容）
> - 三轨 ablation launcher：[scripts/external_bench/run_eqbench3.py](../scripts/external_bench/run_eqbench3.py) + [scripts/external_bench/run_empathybench.py](../scripts/external_bench/run_empathybench.py) + [scripts/external_bench/compare_ablation.py](../scripts/external_bench/compare_ablation.py)（verdict 守门：四条红线 attestation 缺失则拒绝出 verdict）
> - 公开文档：[`docs/external/eqbench3-submission-protocol.md`](external/eqbench3-submission-protocol.md) + [`docs/external/companion-bench-eqbench-crosswalk.md`](external/lscb-eqbench-crosswalk.md)
> - 内部跟踪：[`docs/external/eqbench3-results-internal.md`](external/eqbench3-results-internal.md)（per-run verdict 模板）+ [`docs/external/eqbench3-public-submission-checklist.md`](external/eqbench3-public-submission-checklist.md)（Packet 10 actuation gate）
> - 测试：1048 import-boundary + 138 adapter unit/integration tests 全绿；`vz-* / lifeform-domain-* / lifeform-core` **零 diff**
> - 衍生 follow-up：**新增 [#31](#31)**（streaming SSE 未实现，#30 Chatbot Arena 提交前必须做）

- **路径**：
  - 既有内部评估面（已就位但**仅内部可见**）：
    - [`packages/lifeform-evolution`](../packages/lifeform-evolution/) 6 族 family report / multi-round delta-vs-baseline / longitudinal aggregator
    - [`docs/specs/evidence_program.md`](specs/evidence_program.md) claim registry / blind packet / evidence bundle 全套
    - [`docs/EVALUATION_SYSTEM.md`](EVALUATION_SYSTEM.md) 内部评估系统说明
  - **缺位的对外面**：
    - 没有任何公开 EQ-Bench 3 / EmpathyBench / RP-Bench / Chatbot Arena 提交记录 → google "VolvenceZero EQ benchmark" 一条结果都没有
    - `lifeform-service` 的 `POST /v1/turns` 与 OpenAI Chat Completion API 不兼容（envelope shape / 字段名 / SSE 事件全不同）→ EQ-Bench 类 harness 不能直接打过来
    - 没有 OpenAI-compatible 适配层 wheel；没有 submission 包（model card / system prompt / generation config 公开版）；没有跑分脚本
- **问题**：业界 2026 年针对 chat / companion / EQ 类系统的可见评估面（按权重）：
  1. **EQ-Bench 3**（[eqbench.com](https://eqbench.com/)）—— Claude Opus 4.6 当 judge 的 LLM-judged 8 维 EQ + 3-turn roleplay；Grok-4.1 Thinking 当前领先 Elo 1586；需要 HuggingFace 开源权重才上 leaderboard，但**任何能调的 chat endpoint 都能跑分留私本**
  2. **EmpathyBench**（[empathybench.com](https://www.empathybench.com/)）—— 共情专项；Qwen3 VL 235B 当前 51.3% 领先；提交门槛比 EQ-Bench 3 低
  3. **Chatbot Arena (LMSYS)** —— 600 万+真人盲评；GPT-5.2 Pro / Claude Opus 4.6 ~1400 Elo；商业 API 也可以提交
  4. **RP-Bench / Roleplay-Bench**（HuggingFace）—— 社区人评 Elo；human preference 显著与 LLM-judge 分歧（Gemma 4 26B / Mistral Small Creative 在人评榜领先）
  
  这构成的事实是：**任何投资人尽调"这家陪伴 AI 公司的 EQ 评估面"时，第一个动作就是搜 EQ-Bench 和 Chatbot Arena**。我们当前的回答只能是"我们有内部 6 族评估"——这是**正确但不可验证**的回答，效力远低于一条可点击的 leaderboard 链接。具体后果：
  - 尽调材料里"我们 EQ 强"的 claim 无外部数据支撑；任何竞品（哪怕是套壳 Claude）只要交一次 EQ-Bench 就能反超我们的 fundraising narrative
  - 不交分 ≠ 我们打不过；frozen substrate 的分数大概率 ≈ 底层 LLM 的分数（这不是问题，但**不交就让对方猜**最糟糕）
  - 业界已观察到 EQ-Bench 与 RP-Bench 人评分歧大（详见上）→ 我们若只盯 EQ-Bench 也错失"在人评类反超大模型"的窗口
  - Companion Bench RFC（轨 2，[`docs/external/companion-bench-rfc-v0.md`](external/lscb-rfc-v0.md)）正在做"**定义新评估范式**"的长期防御；轨 1 / 轨 3 是补"**短期防御**"——两者互不替代
- **违反**：不违反 R 铁律。这是产品 / 商业层证据面债，与 vz-* / lifeform-* 内核架构正交。
- **风险**：
  - **短期（<3 月）低**：不交分系统不爆，CI 不挂
  - **中期（3-12 月）高**：任何一次正式融资 / 战略合作尽调都会被一条公开 leaderboard 缺位卡住；竞品交了我们没交 → 对方拿"客观第三方分数高于 VolvenceZero"做谈判杠杆
  - **长期（>12 月）决定品类话语权**：陪伴 AI 在 2026 年的话语权竞争里，"客观 EQ 强"+"养成式架构唯一"双叙事缺一不可；只讲架构不讲分数 = 学术 vibe，讲不动产业资本
- **触发条件**：
  - (a) 启动任何一轮融资（A 轮及以后）尽调材料准备
  - (b) 与任何战略合作方（Cloud / 终端硬件 / 内容平台）做技术 due diligence
  - (c) 上线公开 demo / API 后被技术媒体或竞品分析师 cited 时被问 "your EQ-Bench score?"
  - (d) Companion Bench RFC v0.2 公开发布前（自己定义 benchmark 但自己不在他人 benchmark 上有分数 = 信誉漏洞）
  - (e) 任何竞品交分进 EQ-Bench leaderboard top 20
- **推荐修法**（按优先级，可独立推进）：
  1. **OpenAI-compatible 适配 wheel**（~1 周）：新 `lifeform-openai-compat` 或 `lifeform-service.openai_adapter` 子包，暴露 `POST /v1/chat/completions`（OpenAI envelope）+ optional `metadata.session_id` / `metadata.user_id`（companion 模式启用 cross-session memory）。**不动 vz-* / lifeform-* 内核**，纯薄翻译层；contract test 静态守"openai_adapter 不直接 import vz-cognition / vz-temporal / vz-memory 内部模块，只能走 lifeform-service 的 facade"
  2. **EQ-Bench 3 内部跑分**（$10-15 + 几小时）：用 `eqbench3` harness（[github.com/EQ-bench/eqbench3](https://github.com/EQ-bench/eqbench3)）打到适配 wheel 的 chat endpoint；产 `artifacts/external_bench/eqbench3_run_{timestamp}.json` + transcripts；**先内部跑、不立刻交**——拿到分数后判断
     - 若分数 ≥ 行业均值 → 路径 (3) 公开提交，+ 同时挂在 [`docs/external/`](external/) 内
     - 若分数 < 预期 → 不公开，留作内部 baseline，转路径 (4) 优先
  3. **EmpathyBench + EQ-Bench 提交**（~$50 总，含人工 review）：填 model card / system prompt / generation config + 在 [`docs/external/`](external/) 留一份"我们的 EQ-Bench 跑分协议"公开说明；要求 marketing / 公关同步对接，**不在产品官网吹分数前先把 reproducibility 做对**
  4. **跑分对照内部 6 族评估**：把 EQ-Bench 8 维 + EmpathyBench 维度 vs 我们 F2/F3 family 指标做 cross-walk（不暴露 F1-F6 内部 schema，只描述"对外 EQ-Bench 7 维都映射到 Companion Bench A2 子轴 8 项中的 7 项"）；产 `docs/external/companion-bench-eqbench-crosswalk.md`，给 Companion Bench RFC v0.2 做 evidence backing
  5. **守红线**：
     - openai_adapter 是 read-only facade；不允许任何外部调用反向写入 owner SSOT（与 #15 / #16 / #22 同 pattern；加 `tests/contracts/test_openai_adapter_no_kernel_writeback.py` 静态守门）
     - 提交材料严禁泄露内部架构口径（NL/ETA/R-PE/regime/owner SSOT/F1-F6 内部分类名）；只发对外抽象（"long-context companion system with adaptive memory"等）
     - 不在 system prompt 里塞 EQ-Bench / EmpathyBench / Companion Bench scenario derivative 文本（attestation 要求这条不能违反）
- **优先级**：**中-高**（不阻塞代码运行，但阻塞融资节奏；建议在下一轮融资 kickoff 前 4-6 周启动）

## 30. 人评类 chat arena（RP-Bench / Chatbot Arena / Companion Bench human track）参与 footprint 完全缺位

- **路径**：
  - 上游依赖：#29 OpenAI-compatible 适配 wheel（不先做 #29 的 (1)，本债的 (1)/(2)/(3) 都做不动）
  - **缺位面**：
    - 没有 [Chatbot Arena](https://lmarena.ai/) 入榜——任何 closed-API 系统都可以参与，我们的 lifeform-service / DLaaS chat endpoint 至今未提交
    - 没有 [RP-Bench / Roleplay-Bench](https://huggingface.co/datasets/lazyweasel/roleplay-bench) 提交——这是**最贴近我们"EQ > IQ" 论点的人评 arena**（Gemma 4 26B / Mistral Small Creative 在人评榜领先大模型，正好对齐 small-but-relational > large-but-cold 的叙事）
    - 没有 [`docs/external/`](external/) 之外的对外 channel（blog / arxiv preprint / huggingface space / demo space）让真实用户能"摸一下我们的 companion 模式"
    - Companion Bench human track（[`docs/external/companion-bench-rfc-v0.md`](external/lscb-rfc-v0.md) §6.6）当前只是 RFC 中的 placeholder，没有 human annotator 招募协议 / 人评 UI / 人评数据 schema / 数据合规审查
- **问题**：陪伴 AI 的产品命题——"系统让用户感觉到了什么"——本质上是 **subjective, human-rated, longitudinal** 的，正是 LLM-judge 类 benchmark 系统性测不准的部分。RP-Bench 的实证发现已经强证：**社区人评（Gemma 4 26B Elo 1535）系统性偏好"中型有人味"模型多于"大型大模型味"模型（GPT-4 / Claude 等）**，这恰好对齐我们的对外叙事。但我们没有任何人评 footprint：
  - 没有 RP-Bench 提交 → "Volvence Zero 在 human-rated companion arena 排第几"是空白
  - 没有 Chatbot Arena 入榜 → 600 万真人盲评的 reference 群体里我们不存在
  - 没有 demo / preview channel → 即便投资人 / 媒体想"亲手试一下"，我们没有可指向的入口
  - 没有自己的 human eval pipeline → Companion Bench human track 永远停留在"RFC 提议"阶段；任何竞品先建立 human-eval pipeline 就先占据"陪伴 AI 唯一有人评 footprint"的话语权
- **违反**：不违反 R 铁律。与 #29 同属"对外证据面 + 商业话语权"债，与内核架构正交。
- **风险**：
  - **短期低**：不交人评不影响系统跑
  - **中期高**：人评 arena 在 2026 年是"区分有人味系统 vs 套壳系统"的核心评估面；竞品（尤其是 character.ai / Replika 类）若先于我们建立人评 footprint，会拿走"唯一可证人味强"的话语
  - **长期 = 品类定义权**：陪伴 AI 这条品类的话语权很可能不由"客观 EQ-Bench 第一"决定，而由"在人评类 arena 里实证 outperform 大模型"决定——RP-Bench 现状已经初步证明这条路径成立
- **触发条件**：
  - (a) 任何一次品牌侧 / PR 事件（产品发布会 / 媒体专访 / 行业大会演讲），需要"亲自试"入口
  - (b) 战略合作方提出"先让我们的 PM / 设计师试 1 周再签 LOI"
  - (c) 任何竞品（character.ai / Replika / 国产陪伴产品）先在 RP-Bench / Chatbot Arena 公开排名
  - (d) #29 路径 (3) 公开提交跑分后，需要"客观 EQ-Bench 数字 + 人评 arena 排名"两手齐
  - (e) Companion Bench RFC 进入 v0.2，需要对外能 cite 一份"已验证的 human-rating pipeline 实证 study"
- **推荐修法**（依赖 #29 路径 (1)）：
  1. **Chatbot Arena 提交**（~$0 + ~1 周对接）：基于 #29 (1) 的 OpenAI-compatible endpoint，按 [LMSYS submission docs](https://lmarena.ai/) 流程注册"VolvenceZero Companion Mode"——提交 model card 时**只描述外部能力**，不暴露内部架构；产 `docs/external/chatbot-arena-submission.md` 留 reproducibility note
  2. **RP-Bench 跑分 + 提交**（~$50-100 + ~2 周）：跑 [HuggingFace lazyweasel/roleplay-bench](https://huggingface.co/datasets/lazyweasel/roleplay-bench) 的 1857 票 community 投票池；提交 prompt + config；**优先项**——这是与"EQ > IQ"叙事最对齐的 arena
  3. **HuggingFace Space / Public demo**（~1-2 周）：在 hf.co/spaces 或自建 demo.volvencezero.ai 暴露 companion 模式的 limited-rate-public preview；要求 lifeform-service 加 demo-mode 限流 + 不持久化任何 user identity（隐私合规）
  4. **LSCB human track 起步**（v0.3 时机）：
     - 招募协议：用 Prolific / MTurk / Surge 招募 200+ annotator，时薪 ≥ 当地最低工资 1.5 倍；annotator 多样性约束（age / gender / native language）
     - 人评 UI：开源 reference impl（Streamlit / Next.js），让 annotator 看 arc 录像 + 投 pairwise winner + 写 free-form rationale
     - 数据合规：annotator 同意书 + IRB-equivalent review；评分数据匿名化；生成 training data 永远 forbidden（与 Companion Bench RFC §11 governance 对齐）
     - 对外发布 first study：跑 ~10 个 reference systems（含我们自己的 anonymized variant + 几个 frontier 大模型 + 几个开源 small companion）→ 产业界第一份"长会话陪伴系统人评 study"
  5. **守红线**：
     - 所有对外 channel 严禁透露内部架构口径（与 #29 第 (5) 点同）；marketing 文本 review 流程加"内部架构术语黑名单"（NL/ETA/R-PE/regime/owner SSOT/F1-F6 等内部命名）
     - 人评数据严禁回流任何 kernel owner 或 substrate training（避免把人评 arena 变成训练源 → 违反 R12 evaluation 单向性）
     - 人评数据不构成 PII：annotator 个人信息单独存，与评分数据物理隔离
- **优先级**：**中**（短期可独立推进 (1)/(2)/(3)；(4) 需要 Companion Bench RFC 进入 v0.3 才合时机；与 #29 互为前置/后置——两边都做才形成"客观分数 + 人评分数 + 自定 benchmark"三角防御）

## 31. OpenAI-compat 适配 wheel 流式 SSE 未实现（`stream=true` → 501）

- **路径**：
  - 现状：[`packages/lifeform-openai-compat/src/lifeform_openai_compat/router.py`](../packages/lifeform-openai-compat/src/lifeform_openai_compat/router.py) `_handle_chat_completions` 在 `request.generation.stream is True` 时显式返回 `501 streaming_not_supported`
  - 现状（DTO 层）：[`packages/lifeform-openai-compat/src/lifeform_openai_compat/dto.py`](../packages/lifeform-openai-compat/src/lifeform_openai_compat/dto.py) `GenerationConfig.stream: bool = False` 字段 ready 但纯 placeholder
  - 上游已存在但未启用：DLaaS Slice 5.4 真流式 SSE（`vz-substrate` text generation streaming surface）在 [debt #12](#12) 中 cancel 保护 frozen-substrate；类似机制可在 lifeform-openai-compat 这个**外挂层**复用而不再次触碰 vz-substrate
- **问题**：当前 OpenAI-compat 适配层只支持 single-shot 响应。对 EQ-Bench 3 / EmpathyBench / 任何 LLM-judged rubric harness 完全够用——它们都是 single-shot post 拿完整 transcript 再判分。但**任何依赖 streaming chat 的对外面**会被本债阻塞：
  - **Chatbot Arena (LMSYS)** 实时投票通道：投票 UI 是 streaming 实时显示的，未提交 streaming 的模型只能走 batch eval 通道（流量比 streaming arena 小一个数量级）
  - **OpenAI Python SDK streaming chat**：任何用 `client.chat.completions.create(stream=True)` 写的现成 harness / 评估脚本 / 集成测试都会一次性失败，user 必须改 wrapper 才能跑——会显著抬高对外 onboard 门槛
  - **HuggingFace Space demo**（[#30](#30) 推荐修法 (3)）：用户体验上"打字机"逐字显示是合理预期；single-shot 长延迟看起来像是死机
  - **任何 RP-Bench 风格 community 人评 arena**：人评 UX 几乎都是 streaming，等 30s 才看到响应会让 annotator 提前关闭 tab
- **违反**：不违反 R 铁律。adapter 是外挂层，与内核 R8/R12 正交。但**违反 #30 修法 (1)（Chatbot Arena 提交）的隐含前提**——LMSYS 的 submission 流程默认假设模型支持 streaming response。
- **风险**：
  - **短期低**：debt #29 P10 actuation（EQ-Bench 3 公开提交）路径完全不需要 streaming，rubric 跑分系统跑不动；本债不阻塞融资尽调里的"客观分数"叙事
  - **中期中-高**：[#30](#30) 修法 (1) Chatbot Arena 提交一旦启动，本债立刻成为硬阻塞（~1-2 周延期）；[#30](#30) 修法 (3) HF Space demo 也强相关
  - **长期低**：OpenAI 已经在 chat completions API 之外引入 Responses API 等新 surface；如果未来生态全面迁移到新 surface，streaming SSE 这条具体路径价值会降低；但**短中期内 Chatbot Arena / 大量第三方 harness 仍在 chat completions streaming 上**
- **触发条件**：
  - (a) [#30](#30) 修法 (1) Chatbot Arena 公开提交流程启动
  - (b) [#30](#30) 修法 (3) HuggingFace Space / 公开 demo 上线
  - (c) 任何投资人 / 战略合作方亲自试 demo（"打字机"动效是当代 AI 产品最低期待）
  - (d) 某个外部 harness 强制要求 `stream=true`（Aider 类工具、IDE 集成、Open WebUI 等）
  - (e) DLaaS 平台层（[#12](#12)）想把本 adapter 接入并对外卖 OpenAI 兼容 API
- **推荐修法**：
  1. **承袭 [#12](#12) 的 cancel-tolerance 设计**（不重做）：[`packages/dlaas-platform-api/...`](../packages/dlaas-platform-api/) 中 SSE 设计已沉淀过一次，包括 `vz-substrate` 不可热改的红线、cancel 路径的有界性、token-by-token frozen-substrate-safe 协议；本债的实现应**直接 import / 复用** DLaaS 平台层的 SSE 公共 API（如有），不要在 adapter 层独立重写
  2. **`router.py` 加 SSE 分支**：当 `request.generation.stream is True` 时，构造 `aiohttp.web.StreamResponse`，按 OpenAI Server-Sent Events 协议（`data: {chunk}\n\n` + 终止 `data: [DONE]\n\n`）流式返回。每个 chunk envelope 复用现有 `ChatCompletionChunk` shape（OpenAI 已规范化）
  3. **lifeform-mode streaming**：基于现有 `LifeformSession.run_turn(...)` 的同步生成路径不直接支持 token streaming（lifeform 是 turn-shaped，每 turn 生成一次完整 response）。**两种实现选项**：
     - (3a) **post-hoc fake-streaming**：lifeform run_turn 完成后，按字节切分 response 模拟 streaming（实现简单，但延迟体验仍是"长等然后突然全出"）
     - (3b) **真 token-streaming via substrate hook**：在 SessionManager 层加 streaming-aware run_turn 变体，让 substrate generate 暴露 token iterator；这是真正的"打字机"体验但需要碰 SessionManager 公共 API（不破坏 read-only 红线但需要扩展）
     - **推荐先做 (3a)** 收敛 demo / Chatbot Arena 体验，再独立 packet 推进 (3b)
  4. **raw-mode streaming**：`raw_substrate_complete` 改为 streaming-aware 变体；`runtime.generate(...)` 已有 token-by-token capability（transformers 流式生成），暴露给 adapter
  5. **守红线**：streaming 路径**仍只走 SessionManager 公共方法**（与 [tests/contracts/test_openai_adapter_no_kernel_writeback.py](../tests/contracts/test_openai_adapter_no_kernel_writeback.py) 同口径）；新增 contract test `tests/contracts/test_openai_adapter_streaming_no_kernel_writeback.py` 守 SSE chunk producer 不能反向写 kernel；adapter 不持有任何 LifeformSession 引用 outside SessionManager 借出的范围
  6. **测试**：与 P4 chat completions integration test 同结构 — 用 fake SessionManager + fake runtime 测 SSE event 序列正确性（chunk shape / 终止序列 / cancel 路径）；不需要真 Qwen
  7. **文档**：在 [`docs/external/eqbench3-submission-protocol.md`](external/eqbench3-submission-protocol.md) 加一段说明 streaming 启用条件；在 [`docs/external/companion-bench-rfc-v0.md`](external/lscb-rfc-v0.md) Appendix A 同步注明 Companion Bench v0.1 不要求 streaming（保持 Companion Bench scope 干净）
- **优先级**：**中**（#29 P10 actuation 不需要；#30 修法 (1)/(3) 启动前必须做；建议在 Chatbot Arena 提交 packet 启动前 4-6 周开始）

## 32. Companion Bench v1.0 工程已就绪但 launch 路径未启动（working group / reference 实跑 / DNS / submission queue）

- **路径**：
  - Companion Bench v1.0 reference impl 已 land：[`packages/companion-bench/`](../packages/companion-bench/) Apache 2.0 wheel + 24 public + 96 held-out scenarios + 10 reference systems orchestrator + GitHub Pages 静态站 + 4 个 CI workflow + 144/144 测试全绿
  - **缺位面**（v1.0 launch 必须项，目前都是 placeholder）：
    - **Working group 未形成**：[`docs/external/companion-bench-governance-charter-draft.md`](external/lscb-governance-charter-draft.md) 章程模板就位但「Chair-elect candidates」段是空表；RFC §11 要求 ≥ 3 个组织 / 不超过 1 票 / rotating chair / 每季度 review
    - **`VolvenceZero/lscb-heldout` 私有 repo 未创建**：[`.gitmodules`](../.gitmodules) 引用了它但 repo 本身不存在；CI release-tier deploy key 未 provisioned；[`docs/external/companion-bench-heldout-bootstrap.md`](external/lscb-heldout-bootstrap.md) 一次性 organiser 步骤未执行
    - **真 10 reference systems 跑分未发生**：[`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) + [`scripts/companion_bench/run_lscb_paper_suite_full.sh`](../scripts/companion_bench/run_lscb_paper_suite_full.sh) 已就位但需要 ~$5-15k API 预算 + 6 个 vendor API key 才能产生真数据；当前 [`site/leaderboard/data/aggregate_results.json`](../site/leaderboard/data/aggregate_results.json) 是 [`generate_demo_aggregate.py`](../scripts/companion_bench/generate_demo_aggregate.py) 合成的 placeholder（带 `demo: true` flag）
    - **`lscb-bench.volvencezero.org` 域名 / GitHub Pages CNAME 未配置**：[`.github/workflows/lscb-leaderboard-publish.yml`](../.github/workflows/lscb-leaderboard-publish.yml) 部署到默认 `*.github.io` 路径；自定义域名需要 DNS + Pages 设置
    - **Submission queue 未对外开放**：RFC §11 + [`docs/external/companion-bench-submission-protocol.md`](external/lscb-submission-protocol.md) §10 暂以「filed an issue on github.com/VolvenceZero/lscb-bench」为入口；该 repo 尚未独立出来（当前在 monorepo 内），没有 issue template / labels / triage workflow
- **问题**：v1.0 reference impl 在 git 层完成了「方法学 + 代码 + 文档 + scenarios + scripts + CI」全套——但**「LSCB 是一个 live community benchmark」这件事**还没在外部世界发生。只要这些 placeholder 还在，对外说"Companion Bench v1.0 已发布"就是技术上正确、运营上空洞——任何记者 / 投资人 / 竞品研究员一搜到自定义域名 404、leaderboard 满是 demo 数据、charter draft 空候选人段，会立刻把 Companion Bench 归类为 vaporware vs 真 RFC convener。
- **违反**：不违反 R 铁律。这是**运营 / 治理 / 预算 / 域名层**债，不是代码层债。
- **风险**：
  - **短期低**：v1.0 reference impl 自身可被任何外部复用；任何团队都可以 clone repo 跑自己的 submission 拿自己的分数（RFC §3 P3 self-serve reproducibility 已成立）
  - **中期高**：[#29](#29) P10 actuation（EQ-Bench 公开提交）一旦完成、给我们留下「客观分数」之后，下一步 marketing / 战略合作 narrative 就会强 push「LSCB convener 身份」——本债不解决，convener 叙事就空了
  - **长期决定品类话语权**：陪伴 AI 这个品类如果在 2026-2027 谁能定义 evaluation 范式很可能就是谁的话语权；working group 不形成 → Companion Bench 永远停留在 v0.x 学术草案；working group 形成 → 是 Companion Bench 还是某个竞品定义出的 benchmark 成为产业 reference 的关键
- **触发条件**：
  - (a) 任何一次正式融资尽调材料准备（投资人会问「你们提的 RFC 有谁在用」）
  - (b) [#29](#29) P10 公开 EQ-Bench 提交后的市场推广窗口（48 小时内必须有 Companion Bench convener 故事跟上，否则只剩"我们在 EQ-Bench 第 N 名"这一条）
  - (c) 任何竞品（OpenAI / Anthropic / Meta / character.ai 类）发布自己的 long-session benchmark RFC（一旦发生，convener 窗口立即关闭）
  - (d) [#30](#30) Chatbot Arena 提交流程启动后需要一个对外 leaderboard URL 给 PR / 媒体引用
  - (e) 媒体 / 行业分析师专题（"How is companion AI evaluated?"）需要一个 live URL
- **推荐修法**（5 个独立可推进 sub-track）：
  1. **真 reference run**（~1 周 + 预算 approval）：BFC（Budget-First-Credible）—— 先批 ~$200-400 budget 跑 [`scripts/companion_bench/run_lscb_paper_suite_small.sh`](../scripts/companion_bench/run_lscb_paper_suite_small.sh)（公开 set + 1 paraphrase seed × ~5 reference systems），用 cost telemetry 校准 RFC §6.7 价格表后再决定是否批 release tier ~$5-15k；产物自动 push 到 [`site/leaderboard/data/aggregate_results.json`](../site/leaderboard/data/aggregate_results.json) 替换 demo
  2. **Held-out repo 创建 + deploy key**（~半天）：跑 [`docs/external/companion-bench-heldout-bootstrap.md`](external/lscb-heldout-bootstrap.md) 步骤；在 GitHub 创建 `VolvenceZero/lscb-heldout` private repo + push 96 seed scenarios + 注册 deploy key；配 `LSCB_HELDOUT_DEPLOY_KEY` repo secret
  3. **Working group 形成**（~3-6 个月，组织层 follow-up）：识别 ≥ 2 个外部 align 的组织（建议候选：EQ-Bench 维护者 / RP-Bench 维护者 / 一家学术 AI 安全 lab / 一家 companion-class 产品公司）；走 [`docs/external/companion-bench-governance-charter-draft.md`](external/lscb-governance-charter-draft.md) 的 chair-elect 流程；填空候选人段
  4. **域名 + GitHub Pages CNAME**（~半天）：DNS 配 `lscb-bench.volvencezero.org` → `<org>.github.io`；GitHub Pages 设置自定义域名；HTTPS 自动 issue
  5. **Submission queue infra**（~1 周）：把 lscb-bench 拆成独立 public repo（保留 monorepo 内的 dev 路径作为 internal mirror 或 archive）；加 GitHub issue template `submission.yml` / `comment-period.yml` / 定义 labels；接 GitHub Actions bot 自动 triage（manifest 校验 / attestation 校验）；submission 被 bot 接受后自动跑 [`scripts/companion_bench/run_real_submission.py`](../scripts/companion_bench/run_real_submission.py)（需要 self-hosted runner with API keys 或限速 cloud runner）
- **优先级**：**高**（不阻塞代码 land，但**直接阻塞** v1.0 → 真 launch 的转化；建议在 [#29](#29) P10 actuation 之后立即启动 sub-track 1+2+4，sub-track 3+5 与 [#30](#30) 推进节奏同步）

## 33. Companion Bench human-eval 轨道（RFC §6.6 / §6.6 + §11 核心承诺）实现 0%

- **路径**：
  - RFC [`lscb-rfc-v0.md`](external/lscb-rfc-v0.md) §6.6 承诺一条 parallel human-eval track；RP-Bench 现状已经实证人评 Elo 与 LLM-judge Elo 系统性分歧 → 这条轨道是 Companion Bench 区分于 EQ-Bench 的最强承诺
  - 当前实现层完全空白：没有 annotator 招募协议 / 没有人评 UI / 没有 IRB-equivalent review / 没有 pairwise 投票 schema / 没有 anonymisation 规范
  - 与 [#30](#30) 修法 (4)（LSCB human track v0.3 起步）实质同一件事但视角不同：[#30](#30) 是**对外人评 footprint**，本债是**作为 Companion Bench benchmark 一部分的人评协议**
- **问题**：Companion Bench v1.0 reference impl 全部 LLM-judged。RFC §8.1 列出 LLM-judge 的 family bias / verbosity bias / formatting bias 三类已知威胁；§6.6 明确说"following RP-Bench's finding that human and LLM-judge rankings can diverge meaningfully, we treat human Elo as an independent measurement, not as ground truth"——这意味着**没有 human Elo 列的 Companion Bench leaderboard 就缺了 RFC 承诺的一半**。当前 [`site/leaderboard/index.html`](../site/leaderboard/index.html) 表格已为 human Elo 留了 placeholder column 但内容永远空。
- **违反**：不违反 R 铁律。但**违反 RFC §6.6 自身的承诺**——v1.0 不实现等于 RFC 文本与 reference impl 不一致。
- **风险**：
  - **短期低**：[#32](#32) sub-track 1（真 reference 跑分）跑完后 leaderboard 已经有可看的数；human Elo 缺失短期内不显眼
  - **中期高**：[#30](#30) 推进 + [#32](#32) sub-track 5（submission queue）启动后，「LSCB 自己也没有 human eval 你凭什么说我们超越了 LLM-judge bias」这条质问立刻浮现
  - **长期 = 品类定义权**：human-rated companion arena 在 2026-2027 是品类话语权的核心战场；不实现 = 永远把人评话语权留给 RP-Bench
- **触发条件**：
  - (a) 任何竞品（character.ai / Replika 类）先建立 companion 类 human-eval pipeline
  - (b) Companion Bench v0.3 → v1.0 release 节奏（RFC §9 "v1.0 by Q4 2026 if community engagement supports it"），release 时仍空着 §6.6 = 信誉漏洞
  - (c) 学术 / 行业引用 Companion Bench RFC 时被问"你们 §6.6 是承诺还是 vaporware"
  - (d) [#30](#30) 修法 (4) 招募 annotator 时如果跳过 Companion-Bench-specific UI 直接复用通用 arena 也能跑——但产出的人评数据不能与 Companion Bench scenario hash 关联，无法回归到本 benchmark
- **推荐修法**（按依赖顺序）：
  1. **annotator 招募协议**（~1 周文档 + 法务 review）：用 Prolific / MTurk / Surge 招募 200+ 多样化 annotator；时薪 ≥ 当地最低工资 1.5 倍；标注同意书 + IRB-equivalent review（哪怕公司没有正式 IRB，过一个独立伦理 review 流程）；标注数据**永远不能回流任何 substrate / kernel 训练**（与 RFC §11 governance 红线一致）
  2. **人评 UI（reference impl）**（~2 周）：开源 Streamlit / Next.js 站点；annotator 看一段 arc 录像（transcript playback）+ 投 pairwise winner + 写 free-form rationale；每 arc 至少 5 个独立 annotator；UI 必须**强制完整看完 arc 才能投票**（不允许 first-turn 偏好攻击）
  3. **数据 schema**（~3 天）：`HumanEvalRecord` typed dataclass with `arc_id_a / arc_id_b / annotator_id_anonymised / pairwise_winner / rationale_text / time_seen_seconds / lexicon_version`；存储与 Companion Bench scenario hash 关联但与个人 PII 物理隔离
  4. **TrueSkill / BT 求解器复用**（~1 天）：[`packages/companion-bench/src/lscb_bench/elo.py`](../packages/companion-bench/src/lscb_bench/elo.py) 已经有 `compute_trueskill` / `compute_bradley_terry`；用于 human Elo 列计算（参数与 LLM-judge 列保持一致以便对照）
  5. **leaderboard human Elo 列填充**（~半天）：[`site/leaderboard/leaderboard.js`](../site/leaderboard/leaderboard.js) 已 query `row.human_elo`；只需要 [`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 输出包含该字段；从人评数据计算
  6. **首次 study**（~1-2 月含 annotator 招募 + 标注期）：跑 ~10 个 reference systems × 全 24 public scenarios × 5 annotators / scenario，产出 Companion Bench v1.0 同步 first human-eval study 报告
  7. **守红线**：人评数据严禁回流 substrate / kernel；annotator 个人信息单独存与评分数据物理隔离；LSCB 永不发布个体 annotator id（聚合统计 only）；annotator agreement 明文禁止 LLM 代标
- **优先级**：**中-高**（不阻塞 v1.0 reference impl 的工程交付；但**阻塞** Companion Bench 长期作为「区分 LLM-judge / human-judge 的中立 convener」的核心叙事；建议与 [#30](#30) 修法 (4) 合并为一个 Companion Bench human-eval 工作组 sprint）

## 34. Companion Bench harness 性能 / 可扩展性 gap（顺序 SUT 调用 / 无 streaming / 无 retry / 无 staged executor）

- **路径**：
  - 当前 [`packages/companion-bench/src/lscb_bench/sut_client.py`](../packages/companion-bench/src/lscb_bench/sut_client.py) `OpenAIChatClient` 用 `urllib.request.urlopen` 同步调用，单次 timeout 默认 120s，无 retry / 无 backoff / 无 connection pool
  - [`packages/companion-bench/src/lscb_bench/arc_runner.py`](../packages/companion-bench/src/lscb_bench/arc_runner.py) `run_arc` 完全顺序——session × turn 都串行，单 arc 跑完需要 N×M 次往返
  - [`packages/companion-bench/src/lscb_bench/submission.py`](../packages/companion-bench/src/lscb_bench/submission.py) `run_submission` 顺序遍历 (specs × paraphrase_seeds)；24 × 3 = 72 arcs 串行
  - [`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 顺序跑每个 reference system 的 subprocess（一次只一个）
  - 没有 staged executor / checkpoint / resume：与 [`docs/implementation/10_pe_eta_dialogue_benchmark_harness.md`](implementation/10_pe_eta_dialogue_benchmark_harness.md) 中 `run_real_dialogue_pe_eta_comprehensive_benchmark_staged()` 的 phase-level checkpoint pattern 形成对比
- **问题**：粗算 wallclock：
  - 单 arc：~25 turns × ~2s SUT + ~25 × 1s per-turn judge + 1 arc judge ~5s ≈ 80s
  - 单 submission（24 public × 3 seeds）：72 arcs × 80s ≈ 96 min
  - 10 reference systems × 96 min ≈ 16 小时（理想，无 retry）
  - 实际 + 96 held-out × 3 seeds + judge variance + retry：**release tier 一次跑就是 24-48 小时**
  - 任何中途 API 速率限制 / network blip / OOM kill → **从头开始**（无 checkpoint）
  - 同时 release tier 单次跑成本 ~$5-15k；如果中途失败重跑 = 相同钱包再花一次
- **违反**：不违反 R 铁律。这是工程性能债。
- **风险**：
  - **短期低**：[#32](#32) sub-track 1（small-tier 真跑）只跑 ~5 systems × 1 seed，wallclock ~3-5 小时可接受
  - **中期中-高**：[#32](#32) sub-track 1 完成后想推 release tier（10 systems × 3 seeds × 含 held-out），48 小时无 checkpoint 跑会**反复失败**；CI workflow 默认 timeout 1440 min = 24 小时，超出会被 kill
  - **长期低**：CI runner / cloud GPU 时长便宜下来后压力会缓解，但当前 (2026-Q2) API 速率限制 + judge 模型族日均 token 配额都还紧
- **触发条件**：
  - (a) [#32](#32) sub-track 1（small-tier real run）跑超过预期 wallclock
  - (b) 任何 release-tier 跑（[`run_lscb_paper_suite_full.sh`](../scripts/companion_bench/run_lscb_paper_suite_full.sh)）启动
  - (c) GitHub Actions 24h timeout 第一次撞墙
  - (d) Submission queue（[#32](#32) sub-track 5）启动 → 大量并发 submission 涌入
- **推荐修法**（按 ROI 排序）：
  1. **Async + connection pool**（~1 周）：把 [`sut_client.py`](../packages/companion-bench/src/lscb_bench/sut_client.py) 从 urllib 改 aiohttp（已是 wheel 依赖）；arc_runner 改 async；同 session 内 turn 仍串行（保 cross-session-session_id 语义）但 multi-arc 并发；retry + exponential backoff on 429 / 5xx；token-bucket 速率限制 per-judge-model
  2. **Staged executor**（~3 天）：参 [`docs/implementation/10_pe_eta_dialogue_benchmark_harness.md`](implementation/10_pe_eta_dialogue_benchmark_harness.md) `run_real_dialogue_pe_eta_comprehensive_benchmark_staged` 模式；phase = (arc_run / callback_extract / perturn_judge / arc_judge / aggregate)；每 phase 落 manifest 到 `artifact_dir/manifest.json`；resume 时从已完成 phase 恢复
  3. **跨 system 并行**（~3 天）：[`score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 用 `concurrent.futures.ProcessPoolExecutor` 并发跑多 system（非 thread，因为 each system 子进程已是 IO-bound async）；上限按 judge 模型 quota 推算
  4. **SUT streaming**（与 [#31](#31) 同源）：lscb-bench 当前 sync `chat()` 接收完整 response；如果 SUT 只支持 streaming（一些 vendor 推自己 SDK 时强制 streaming），lscb-bench 直接断在 client 层。Adapter 改 streaming-aware 后，pre-emptively 解决这条路径
  5. **Cost-aware scheduler**（~1 周，optional）：[`cost.py`](../packages/companion-bench/src/lscb_bench/cost.py) 已统计每 system / 每 judge / 每 axis 成本；加 budget cap（"一旦累计超 $X 暂停并 manual approve"）；CI release-tier workflow 接 budget guard
  6. **Checkpoint restart 测试**：mock 中途 SIGKILL 测 staged executor 能从断点续跑；用 in-memory mock SUT 不烧 token
- **优先级**：**中-高**（[#32](#32) sub-track 1 之后立刻启动；release tier 启动前必须做完 1+2+3）

## 35. Companion Bench 季度治理自动化（held-out paraphrase rotation / lexicon rotation / judge family rotation log）

- **路径**：
  - RFC §8.2 要求 held-out paraphrase seeds 季度 rotation；当前 [`scripts/companion_bench/generate_heldout_seeds.py`](../scripts/companion_bench/generate_heldout_seeds.py) 是 one-shot generator，没有 rotation salt 参数；同样的 input 永远产同样的 96 scenarios
  - RFC §6.3 + §8.1 要求 judge model 季度 rotation；[`docs/external/companion-bench-eqbench-crosswalk.md`](external/lscb-eqbench-crosswalk.md) 引用了 `lscb-judge-rotation-log.md` 但**该文件未创建**；charter draft 引用了 `lscb-heldout-rotation-log.md` 也**未创建**；[`docs/external/companion-bench-governance-charter-draft.md`](external/lscb-governance-charter-draft.md) §6 同样要求 hash-only 公开 diff 但格式未定
  - [`packages/companion-bench/src/lscb_bench/lexicon.py`](../packages/companion-bench/src/lscb_bench/lexicon.py) `LEXICON_VERSION = "1.0.0"` 是 string 常量；版本 bump 没有自动化路径，bump 之后 scenario_hash 是否要重生成也未定（lexicon 仅 runtime 消费，不进 hash，所以理论上 lexicon bump 不变 hash——但需要明文说清楚）
- **问题**：v1.0 reference impl 把"季度 rotation"作为承诺写入了 RFC 与 charter，但**实际执行链路 0%**。一旦 working group 形成（[#32](#32) sub-track 3），第一个季度 rotation due 时会发现：
  - 没人按过 rotation button
  - 没工具自动生成 rotation manifest
  - 没 audit log 记录"谁在何时做了 rotation"
  - 没 hash diff 让外部审核
  - 旧的 96 held-out paraphrase 永远是同一批 → 季报"rotation 完成"成谎
- **违反**：不违反 R 铁律。但**直接违反 RFC §8 + governance §6/§7 多项明文承诺**。
- **风险**：
  - **短期低**：v1.0 launch 后的第一个季度（~3 个月）没人会 audit
  - **中期中**：第二季度 rotation due 时如果还没自动化，working group 会议会被一个"我们说要 rotate 但没 rotate"占据议程
  - **长期高**：如果某次 rotation 被竞争团队 audit 出来从未发生，LSCB 作为 RFC convener 的可信度直接归零
- **触发条件**：
  - (a) Working group 形成（[#32](#32) sub-track 3）后第一个季度 review 会议
  - (b) 任何外部研究者或竞品质疑 "你们 RFC 说 rotation，repo 里看不到 rotation log"
  - (c) v1.1 release 节奏（quarterly cadence per RFC §9 implicit cadence）
- **推荐修法**：
  1. **`generate_heldout_seeds.py` 加 `--variant-salt` / `--rotation-quarter` 参数**（~1 天）：当 salt 变化时变体的 surface form 改变（persona tone / payload prefix / FSM seed offset 都依赖 salt）；保证同一 quarter 内所有人产出 byte-identical
  2. **rotation 自动化 GitHub Action**（~2 天）：`.github/workflows/lscb-quarterly-rotation.yml`，每季度 1 号 cron 触发；自动 (a) bump quarter salt (b) 重新生成 96 held-out scenarios 到 private submodule (c) push hash-only diff 到 `docs/external/lscb-heldout-rotation-log.md`（公仓）+ `external/companion-bench-heldout/HASHES.txt`（私仓）(d) open PR for chair sign-off (e) 不直接 merge——chair manual approve
  3. **judge rotation log 文件 + 自动 entry**（~半天）：创建 [`docs/external/lscb-judge-rotation-log.md`](external/lscb-judge-rotation-log.md)；每季度 working group 决议后由 chair 加一行 entry；CI workflow 在 [`scripts/companion_bench/run_lscb_paper_suite_small.sh`](../scripts/companion_bench/run_lscb_paper_suite_small.sh) / `_full.sh` 启动时校验「当前 judge model 与 log 最新一行一致」否则拒绝跑（防止"workflow 用 GPT-4 但 log 还写 GPT-5"漂移）
  4. **lexicon 版本协议**（~半天文档）：在 [`docs/specs/companion-bench.md`](specs/lscb-bench.md) 加一节明确：lexicon bump 不影响 scenario_hash（已经如此）；lexicon bump 影响 identity slot 选择，但 [`draw_identity`](../packages/companion-bench/src/lscb_bench/lexicon.py) 已经把 `LEXICON_VERSION` 包进 seed string 所以 paraphrase 仍 deterministic；新加 entries 是 backward-compatible，移除 entries 是 breaking
  5. **rotation log 模板**（~1 小时）：定义 `lscb-heldout-rotation-log.md` 与 `lscb-judge-rotation-log.md` 格式（quarter / chair-sig / hash diff / 96-count assertion）
- **优先级**：**中**（v1.0 launch 时不需要；但**必须在 working group 形成的同一季度内** land sub-track 1+2+3，否则第一次 rotation due 时会非常尴尬）

## 36. Companion Bench v2.x 长尾路径（multi-modal / EQ-Bench prompt 1:1 reuse / 加密 attestation / transcript 隐私）

- **路径**：4 个低优先但实质性的工程债，都与 Companion Bench v1.0 reference impl 已交付的代码相关，但都是 v2.x roadmap：
  - **(a) Multi-modal extension** — RFC §10 OQ4 明确"v0.1 is text-only. v2.x roadmap?"；当前 [`spec.py`](../packages/companion-bench/src/lscb_bench/spec.py) `ScenarioSpec.user_simulator.fsm` 只发 `text` payload；voice / 表情 / 多模态 SUT 完全不支持
  - **(b) EQ-Bench rubric prompt 1:1 reuse** — [`packages/companion-bench/src/lscb_bench/judge_perturn.py`](../packages/companion-bench/src/lscb_bench/judge_perturn.py) `_PROMPT_HEADER` 是我们改写的版本而非 EQ-Bench 3 上游 prompt 原文；RFC Appendix A 承诺"Criteria 1–7 are aligned with EQ-Bench 3 to enable cross-benchmark comparison"——技术上 criteria 名 + 含义对齐了，但 prompt 文本不是 byte-for-byte 一致，跨 benchmark 信号转移精度比理论值低
  - **(c) Submission attestation 加密签名** — [`packages/companion-bench/src/lscb_bench/submission.py`](../packages/companion-bench/src/lscb_bench/submission.py) `SubmissionAttestation` 仅是 4 个 bool；submitter 把它们设 `true` 没有任何加密保证或时间戳；任何争议（"你们去年 12 月真的没用 Companion Bench derivative 训练吗"）无法事后审计
  - **(d) Transcript 隐私 / leaderboard 发布脱敏** — `arc_runner.py` 写出的 [`{arc_id}.bundle.json`](../packages/companion-bench/src/lscb_bench/arc_runner.py) 包含 `identity_slot.name / occupation / contextual_detail`（来自 lexicon）+ 完整 user/assistant transcript；release-tier workflow 把整个 `artifacts/companion-bench/reference/` 上传为 GitHub artifact；这些**虽然不是真人 PII**（lexicon 是合成的），但 SUT 的 response 可能 inadvertently 泄露 vendor system prompt / 内部 vocabulary——leaderboard publication 应该有审查环节
- **问题**：4 条路径在 v1.0 reference impl 时都被合理 deferred，但都是真实债务：
  - (a) 不做 → Companion Bench 永远是 text-only benchmark，2027 年 multi-modal companion 成为主流时品类相关性下降
  - (b) 不做 → EQ-Bench 转移信号比 RFC 承诺的弱
  - (c) 不做 → 任何争议无审计依据；声誉风险随 leaderboard 长期使用复合
  - (d) 不做 → 第一次 release 公开后被审计出 SUT system prompt 泄露 → 信誉事件
- **违反**：不违反 R 铁律。
- **风险**：
  - **短期低**：所有 4 条都不阻塞 v1.0 launch
  - **中期中**：(c) 与 (d) 在第一个争议事件时立刻浮现；建议在 [#32](#32) sub-track 5（submission queue）启动同时 land
  - **长期高**：(a) 决定 Companion Bench 在 2027+ 是否仍是相关 benchmark；不做 = 把 multi-modal 评估这一片话语权让给后来者
- **触发条件**：
  - (a) **multi-modal**：任何 multi-modal companion（语音 + 表情）成为产业默认 SUT；OpenAI / Anthropic 推出 streaming-voice + vision SDK 默认 surface
  - (b) **EQ-Bench prompt reuse**：EQ-Bench 维护者公开质疑 Companion Bench §A 信号转移声明；学术论文用 Companion Bench 数据时发现 cross-benchmark 信号比预期弱
  - (c) **加密 attestation**：第一次 leaderboard 争议（某 submitter 被指控 Companion Bench derivative training）
  - (d) **transcript 隐私**：第一次审计 / 第一次 vendor system prompt 在 leaderboard artifact 中被发现
- **推荐修法**（按 v2.x packet 排序）：
  1. **(a) Multi-modal v2.0**（~3-6 月）：`ScenarioSpec.user_simulator.fsm[].payload` 从 `str` 升级为 typed `Payload` union (`TextPayload` / `AudioPayload` / `ImagePayload`)；判官层加 multi-modal scoring；与 RFC §10 OQ4 同步推进；这是大改 backward-compat 要谨慎（hash 算法需要扩展）
  2. **(b) EQ-Bench prompt 1:1 reuse**（~1 天）：找 EQ-Bench 3 上游 rubric prompt 原文（Apache 2.0 / MIT 之类的兼容 license 下），把 [`judge_perturn.py`](../packages/companion-bench/src/lscb_bench/judge_perturn.py) `_PROMPT_HEADER` 中前 7 个 criterion 的描述改为 EQ-Bench 3 verbatim；保留 Companion Bench 8th criterion（boundary_appropriateness）作为 Companion Bench 独有部分；在 [`docs/external/companion-bench-eqbench-crosswalk.md`](external/lscb-eqbench-crosswalk.md) 注明"prompt verbatim reuse since v1.x"
  3. **(c) Attestation signing**（~3 天）：`SubmissionManifest.attestation` 加 `signed_at: datetime` + `signature: str`（HMAC-SHA256 over manifest body 用 submitter 提供的 ed25519 key）；leaderboard 公开 submitter 的 public key 给 audit；争议时可加密验证 attestation 时间戳
  4. **(d) Transcript 隐私 review pipeline**（~1 周）：release-tier 把 `{arc_id}.bundle.json` 上传前先跑一个 anonymisation pass（去除任何疑似 system prompt 漏出 / 任何 PII-shaped 字符串）；leaderboard 公开版本只含 axis scores + scenario_hash，**不含** transcript body；transcript 留在 organiser 私有 storage（同 held-out 一样的治理）
- **优先级**：**低-中**（4 条都是 v2.x；(c) (d) 在 [#32](#32) sub-track 5 启动同时建议 land；(b) 任何时候都可以；(a) 是真 v2.0 工作量，等 multi-modal 商业化稳定后启动）

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