# Known Architecture Debt

> Status: tracked, not blocking
> Last updated: **2026-05-29** (DLaaS scale/isolation/rare-heavy packet — per-end-user identity safety, rare-heavy training executor, multi-process/multi-GPU pod routing; advances #17 / #46 / #69 / #76 + rare-heavy executor)

> 2026-05-29 update v2 (DLaaS scale + isolation + rare-heavy executor packet — P0/P1/P2 全实现, backward-compatible/opt-in): 把"负载均衡/substrate 分配 / rare-heavy 调度 / digital-life 隔离 / end-user id 管理"四块从评估出的缺口一次性补齐,默认行为不变,新能力 env opt-in。**P0 per-end-user 身份安全**:(a) session 复用一致性守门 —— `_get_or_create_session` 复用同 `session_id` 但 `end_user_ref` 不一致时返回 `409 session_end_user_mismatch`(默认开,`VZ_ALLOW_SESSION_END_USER_REMAP=1` 关;DLaaS + OpenAI-compat 两路都接;堵住跨用户内核状态泄漏);(b) 两层 `tenant:end_user` scope —— `tenant_id` 经 adopt→`InstanceManager.acquire`→`SessionManager` 线程化,`VZ_TWO_LAYER_SCOPE=1` + `scope_strategy=="tenant_ai_end_user"` 时走 `bind_session` 两层(默认仍单层,避免重键现有 closed-alpha 磁盘 memory);(c) per-ai_id memory root —— `VZ_PER_AI_MEMORY_ROOT=1` 时 `acquire` 算 `{base}/{tenant}/{ai_id}` 根目录传给 SessionManager(两 ai_id 不再共享磁盘 memory)。**P1 rare-heavy 平台执行器**(补 [#17 评估] 指出的"平台无 executor"缺口):新 `training_jobs` 持久表([`dlaas-platform-registry/training_jobs.py`](../packages/dlaas-platform-registry/src/dlaas_platform_registry/training_jobs.py),schema v7 additive)+ `TrainingJobExecutor` 队列 + 后台 worker(`VZ_TRAINING_WORKER=1` 启用)推进 `pending→running→succeeded/failed`,可插拔 runner(`synthetic` 默认 / `figure_lora` 受门控);job create 强制 `allow_adapter_training` / `allow_rare_heavy_refresh`→`403`;promote 仍需 gate_evidence([`training_executor.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/training_executor.py))。默认 worker 关 → 旧"只建记录不执行"行为不变。**P2 多进程/多 GPU pod 路由**(#17 第二阶段真接线):`LauncherProtocol` 让 api 不再硬 `isinstance(InstanceManager)`;`RemoteInstanceManager`(HTTP 代理,可注入 transport)+ `MultiPodLauncher`(placement sticky 路由 + `forward_interaction` 远程转发 + placement 派生 status/overview)+ `PodProcessSupervisor`/`pod_server`(每 GPU spawn 子进程,`CUDA_VISIBLE_DEVICES`);`_dispatch_envelope_to_instance` 当 launcher 暴露 `forward_interaction` 时把整个 envelope RPC 转发给属主 pod(否则本地);`substrate_profile_id`+`runtime_backend` 经 `build_runtime_for_profile` 真选 transformers/vLLM runtime,adopt 时校验 backend 不匹配→`409`;`build_dlaas_app` 在 `VZ_MULTI_POD=1`+`VZ_MULTI_POD_SPECS` 时建 MultiPodLauncher+supervisor,否则默认单 InstanceManager。**测试**:P0 9 + P1 13 + P2 16 + dispatch-forward 2 全绿(fakes/injectable transport/loopback);真 GPU/vLLM 多进程 e2e 仍需运维多卡环境验证(本环境无多 GPU,只 fake/单测)。**仍 SHADOW/待真机**:多 pod 真子进程 spawn + RPC 流式(SSE over RPC)未端到端验证;rare-heavy `figure_lora` runner 的真 PEFT bake 需 GPU+corpus(base 实现 delegate 给运维 override);两层 scope 切换需 memory 迁移说明(不静默重键)。详见 plan [`dlaas scale isolation rareheavy`](../../.cursor/plans/dlaas_scale_isolation_rareheavy_6e7b1db0.plan.md)。

> 2026-05-29 update (DLaaS substrate + LoRA management packet — substrate/lora 加载 · 多用户共享 · substrate 管理收口): 把 DLaaS 平台层 substrate + persona-LoRA 这套机制从"adopt 路径丢 checkpoint / profile 仅元数据 / 全局池 last-register-wins / 无并发多 LoRA / 单进程单 substrate"一次性推进到可用。**已 land**：(a) **PEFT-through-adopt 修复** — `register_bundle_persona_lora` 现透传 `artifact.peft_checkpoint_dir`（之前丢弃 → adopt 静默走 LayerNorm-eaten hook fallback），并经新 `volvence_zero.substrate.resolve_peft_checkpoint_dir` + `VZ_PEFT_CHECKPOINT_ROOT` 在 service CWD ≠ bake CWD 时仍能定位 checkpoint（[`peft_checkpoint_paths.py`](../packages/vz-substrate/src/volvence_zero/substrate/peft_checkpoint_paths.py)）；(b) **substrate profile registry** — `GET /dlaas/v1/catalog/substrate-profiles` 改由 [`substrate_profiles.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/substrate_profiles.py) 提供；adopt 校验未知 profile→400、mode 不匹配→409；**`adapter_policy` 真执行**（`none` 禁 L2 LoRA 激活，线程穿 SessionManager→synthesizer 的 `persona_lora_enabled` 闸）；(c) **per-tenant 作用域池** — 每个 `SessionManager` 持自有 `PersonaLoRAPool`，消除全局池跨租户 last-register-wins 串扰（同 `figure_id` 不同 bundle 的两租户隔离，附并发 evidence test，部分推进 **#61** / **#45**）；(d) **adapter VRAM 缓存** — `activate_peft_adapter` 改为 `VZ_LORA_CACHE_MAX` 有界 LRU 常驻多 adapter（[`peft_adapter_cache.py`](../packages/vz-substrate/src/volvence_zero/substrate/peft_adapter_cache.py)），命中不再每 turn `from_pretrained`，退出 disable 保 R2 行为级冻结基底（real tiny-gpt2 test 守门）；(e) **vLLM 多 LoRA 运行时** — `VLLMOpenWeightResidualRuntime` + `VLLMLoRARouter` per-request `LoRARequest`（[`vllm_runtime.py`](../packages/vz-substrate/src/volvence_zero/substrate/vllm_runtime.py)，`vz-substrate[vllm]` 可选 extra），active checkpoint 走 `contextvars` 按 async task 隔离 → 放松 transformers 的 serial-decode 假设（部分推进 **#61**）；(f) **多 pod placement** — `AiIdPlacementRouter` + `MultiPodLauncher`（sticky / capacity-aware / substrate_profile 约束 / migrate，[`placement.py`](../packages/dlaas-platform-launcher/src/dlaas_platform_launcher/placement.py)），为 **#17** 跨进程/跨 GPU 共享落地 routing 脚手架（pod 进程 spawn 仍是运维注入）；(g) **substrate fingerprint 守门** — adopt 时 bundle `compatible_substrates` 与运行 substrate `model_id` 不匹配→409（substrate-upgrade-protocol）。**仍 SHADOW/待 GPU**：#17 第二阶段 `RemoteResidualRuntime` IPC 真共享 + 真 pod 进程编排；#45 真 Qwen perf 床 30min 负载基线；#61 真 N≥10 GPU 并发 multi-LoRA logits 吞吐基线（本批给出单测级隔离 evidence，未给 GPU 吞吐数）。详见 plan [`dlaas substrate lora`](../../.cursor/plans/dlaas_substrate_lora_8035b246.plan.md)。

> 2026-05-18 update (debt #84 入档 — companion-ref-harness packet H-A SHADOW land + 4 sub-packet 后续追踪): 把 [`docs/moving forward/companion-ref-harness-packet.md`](moving%20forward/companion-ref-harness-packet.md) 5 个 sub-packet 的工程 + 实证 + 方法论 follow-up 一次性入档。**H-A SHADOW 已 land**：新 wheel [`packages/companion-ref-harness/`](../packages/companion-ref-harness/) Apache 2.0 独立协议（与 [`packages/companion-bench/`](../packages/companion-bench/) 并列，互不 import）+ aiohttp OpenAI-compat server + 4 个组件 SQLite store 骨架（H-A 只启用 session_summary，H-B/H-C 表预创建 + 留空守门测试）+ 跨家族 LLM summary extractor（默认推荐 Claude/Gemini 给 GPT-5 写 summary，避开 "GPT-5 self-crib-notes" 批评）+ HarnessPolicy 是 prompt blend 唯一入口（H-B/H-C component 触发 `NotImplementedError` 守门）+ CLI `companion-ref-harness serve --components summary,...`；**105 测试全绿**（70 unit + 35 contract，含 `tests/contracts/test_companion_ref_harness_no_internal_imports.py` 禁 `volvence_zero.*`/`lifeform_*`/`companion_bench.*` + `test_apache_license_header_present.py` 两个 Apache wheel 全文件 license 头守门，跳过零字节 marker `__init__.py`）。**剩余 5 件后续**：(a) H-A ACTIVE (d) 真 ablation evidence（6 substrate × 24 scenario × 1 seed × 2 slice，~$400-600 + 跑分 wallclock）；(b) H-B embed retrieval（BGE-base OSS + sqlite-vec + retrieval block，~2 人周 + ~$500-700 ablation）；(c) H-C user-model + episodic memory（leave-one-out 5 slice ablation 每组件 ≥ 5 分贡献，~2.5-3 人周 + ~$1500-2000）；(d) H-D 方法论文档同步（RFC §7.4 / submission protocol §3 / `reference_systems.yaml` 拆三档 / `run_real_submission.py --reference-systems-set` flag / `archetecture.md` 库清单，~1-1.5 人周 / $0）；(e) H-E competitor case study（Anthropic Memory + OpenAI Assistants 优先，~1-1.5 人周 + ~$200-400）。**与既有 launch 路径的衔接**：H-A → H-C 跑分跟 [#82](#82) phase A.3 共享 6 substrate × 24 scenario transcript cache（合并预算从 $8000 降到 ~$5500）；H-D ACTIVE 必须早于 [#32](#32) sub-track 1 真 reference 跑分启动；H-E `bespoke · case-study` 列**不进 ranked column**，跟 [#29](#29) / [#30](#30) 人评 arena 路径形成显式分工（CompanionBench 跑客观分 / arena 跑 c.ai/Replika/Talkie 等无 API 产品的人评）。**关键 SSOT 守门**：harness wheel 不 import 任何内部包（contract test 强制）；harness 响应**不**携带任何 `x-ref-harness-*` / `x-lifeform-*` / `x-volvence-*` / `x-companionbench-*` 头部（shape-indistinguishable from raw OpenAI-compat endpoint，passthrough 测试守门）。

> 2026-05-15 update v2 (debt #82 / #83 入档 — 与 xfund-pitch-deck-v2.7 同步): 两条 commercialization evidence 缺口正式入档：**debt #82** Companion Bench reference SUT 真跑实证缺位 — 6 大主流 substrate（GPT-5 / Claude Opus 4.7 / Qwen3-Max / DeepSeek V4 / Llama 5 / Gemini）只有 Qwen smoke 已跑且受 #71 / #72 影响 evidence 不可外引，其余 5 个零真实跑分 — leaderboard launch 硬前提（5 phase Phase A.1-A.5 timeline）；**debt #83** 6 JV → in-production 真实 ARR 兑现路径缺位 — deck v2.7 Slide 21 财务全景 + Slide 25 Ask 都依赖"6 JV → 3 in-production · ARR > $1M real"承诺，但当前 0 in-production（3 phase Phase 1-3 timeline）。两条都是融资尽调阻塞条件，列入 #82 与 [`#29`](../docs/known-debts.md) / [`#32`](../docs/known-debts.md) / [`#48`](../docs/known-debts.md) 同源 launch path；#83 与 [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §5 推荐排序绑定。

> 2026-05-14 update (第一性原理违规四轮修复 follow-up — #78-#81 入档): 经过 4 轮针对 R-PE / R8 / R14 / first-principles 的违规扫描 + 修复（累计 **59 处修复 / 跨 27 个文件**：1 轮 13 处 SSOT 吞错误 hasattr / 2 轮 11 处 R-PE evaluation 反向 + R14 regime 字符串硬编码 + R8 跨 wheel 私有访问 / 3 轮 33 处 typed snapshot getattr defense + trace_collector 4 处真 bug 字段名错 + 2 处吞错误 + 1 处早期 return bug / 4 轮 2 处 followup_manager 多候选属性名猜测），仓库主要 R8 SSOT / R-PE 主链 / R14 regime-as-prompt-label 违规已经收尾。第四轮深扫后**剩余 getattr default 绝大多数是有意为之的合规设计**（pure helper 参数 `object | None` / cross-tier duck-typed builder / 第三方 HF API / dataclass `__post_init__` 自验证 / owner 内部 closed schema 遍历）。本批 4 条新 debt 记录前面四轮 plan 中明确 "不在本次范围" 的 4 类 follow-up 候选，等待长期演化触发条件成熟时收口：**#78** framework-agnostic getattr default 系统性 typed-path 升级（pure helpers / cross-tier builders / followup_manager enum 字段） / **#79** `apply_metacontroller_evidence` regime_id 字符串硬编码（owner 内部 reviewer-defined fallback，需 [#44](../docs/known-debts.md) SYS-1 learned policy land 后让位） / **#80** `score_regimes` 用 evaluation 作 input feature（R-PE 严格版本，需 [#48](../docs/known-debts.md) judge sweep 量化偏差后决策） / **#81** `runtime_helpers.py` 6 段 hint summary 文本硬编码（i18n / 新 domain 触发后再 typed 化）。**这 4 条都不阻塞任何 milestone**，列入主要为：(a) 给未来 schema drift / mypy strict / i18n / SYS-1 learned policy land 提供 grep 入口；(b) 不让"reviewer 自我标注 framework-agnostic"的设计选择被时间遗忘；(c) 与已有 [#43](../docs/known-debts.md) Arch Uplift Phase 2 follow-up 节奏对齐。**今天能跑 + 长期影响可演化性**严格符合 known-debts 维护规则。

> 2026-05-14 update (Non-GPU 技术债 sweep — 17 stream / ~30 debts SHADOW v0.2 wiring 集中收口): 把所有不依赖严肃 GPU / 不烧真 LLM API sweep 的纯工程 + spec + 文档债一次性 land 到下一档 SHADOW（每条债都从"scaffold-only"推进到"wiring + injection point + fake-provider contract test 真接通"）。**17 个 stream 全绿**：A1 #46 `bind_session` 默认升双层 + `bind_session_legacy_alias` shim / #47 `substrate_fingerprint` ACTIVE / #49 `MonthlyReportOwner` v0.2 加 `deleted_end_user_count` + `deletion_event_count` + `count_deletion_events_in_window` ledger reader；A2 #45 `realistic_load_*.py` 三方向加 `expected_slo` + `sample_shape` block；B1 #73 SUT subprocess retry + arc-isolation / #74 `SubmissionManifest.system_prompt` + `generation_config` 真注入 `arc_runner` / #75 `CostTracker.record_perturn_judge / record_arc_judge` 真接 (judge 加 `drain_usage_log()`)；B2 #52 / #54 / #56 spec v0 / #55 i18n roadmap + 6 中文 demo scenarios + `ScenarioSpec.language` default `"en"`；B3 #34 `--parallel-sut N` async + `--per-system-retries` / #57 `trusted_runner_skeleton.py` lifecycle stub；C1 #39 `coverage_map` retrieval-augmented floor 修 Wave K Einstein L4 误拒 prod bug / #42 refusal probe N=5 → N=25 (5 domains × 5 题) + ROC re-calibration spec；C2 #58 `figure_refusal_eval.py` / #59 `figure_grounding_eval.py` 加 `--mode fake-judge` 真跑 typed report / #62 `OfflineGateAuditEntry` typed dataclass + per-day jsonl rotation `apply_persona_lora_through_gate(audit_log_dir=...)` 真写；D1 #66 `LLMArchetypeClassifier` 接进 `build_growth_advisor_lifeform` + `GrowthAdvisorLifeformBundle.maybe_classify_archetype(end_user_id, recent_user_turns)` hook + R8 SSOT contract test；D2 #67 `MonthlyReportInputsBuilder` 接 5 个 owner snapshot shim (rupture / boundary / archetype / handoff / protocol_phase) + 30day mock fixture pipeline test + DELETE 后 `deleted_end_user_count` 守门；D3 #64 `growth_advisor_boundary_eval.py` / #68 `growth_advisor_drive_ablation.py` 加 `--mode fake-judge` + Wilson CI + per-archetype matrix；D4 #69 `bind_session(end_user_id=, tenant_id=)` 应用层 surface + #70 handoff queue v0.2 capacity / timeout / cross-restart resume spec + 单进程 N=10 async stress test (`@pytest.mark.perf`)；E1 #13 `DefaultRubricGrader(llm_grader_callable=)` injection + 显式 fallback warning / #14 `EvalStore.upsert_audience_profile(corpus_analyzer_callable=)` enrichment / #15 `_activation_text(asset_fetcher_callable=, template_assets=)` 真抓 corpus；E2 #16 `AffordanceRegistry.set_contract_policy` + `list_for_session(contract_id=)` 真按 contract whitelist 过滤；E3 #12 + #31 `lifeform-openai-compat` SSE streaming (4 frames + `[DONE]` sentinel)；F1 #8 joint_loop owner sharing 加 silent-orchestrator 防御扫描 contract test；G1 #33 Companion Bench human-eval protocol v0.1 + `site/human-eval/` 4 页 (index/apply/training/submit) + #35 `quarterly_rotation.py` 三类自动化 (paraphrase proposals jsonl / lexicon rotation txt / judge family rotation log)；G2 #38 删 7 个 Companion Bench redirect stub (`companionbench-{rfc-v0,governance-charter-draft,submission-protocol,eqbench-crosswalk,heldout-bootstrap,public-scenario-hashes,bench}`) + `verify_site.py` (page/link/data/banner 4 类检查) + `build_site.py --incremental` 跳过未变 submission。**新增 17 个 contract test 文件 / ~80 新测试**；**全 contract suite 跑分 2306 passed / 2 skipped / 3 deselected** (3 deselected 是 pre-existing `vz-cognition-store` / `vz-runtime-owner_hydration_store` / `verification module imports kernel` import-boundary failures，与本 sweep 无关；ReadLints 0 错)。**仍保留 SHADOW 等 GPU/API 批准的 11 条**：#41 真 Qwen-1.5B PEFT bake (1 GPU 月) / #45 perf 床真 30min 负载 (1 GPU 月) / #50 / #61 依赖 #45 / #6 / #7 ≥500 turn 真 trace / #10B item 3 / #10C 真 1.5B Qwen evidence run / #43 / #44 SYS-1 长期训练 / #48 / #71 真 cross-family judge sweep (LLM API ~$80-300) / #72 smoke 切真 hf-shared substrate (GPU) / #29 / #37 EQ-Bench 3 P10 actuation / #30 RP-Bench / Chatbot Arena 公开提交 / #17 DLaaS 跨进程 substrate 共享。完整 stream-by-stream 描述见 [`docs/moving forward/non-gpu_技术债_sweep_b266f4f8.plan.md`](../.cursor/plans/non-gpu_%E6%8A%80%E6%9C%AF%E5%80%BA_sweep_b266f4f8.plan.md) （工作产出物） + commit `26af3c9` "继续修复" (BOSS 累 commit；includes ~62 改 + ~17 新 contract test files)。

> 2026-05-14 update (Einstein figure-as-a-service demo verticals 接入 chat UI — F4.2 → F4.3): 把 `_try_einstein` 拆为 [`einstein-raw`](../packages/lifeform-service/src/lifeform_service/verticals.py) / [`einstein-bundle`](../packages/lifeform-service/src/lifeform_service/verticals.py) / [`einstein-full`](../packages/lifeform-service/src/lifeform_service/verticals.py) 三个 vertical，语义一一对齐 [`PersonaCondition.{RAW,BUNDLE,BUNDLE_LORA}`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/runtime_conditions.py)；新增 [`lifeform_service.einstein_resolver`](../packages/lifeform-service/src/lifeform_service/einstein_resolver.py) 让 `EINSTEIN_BUNDLE_ROOT` / `EINSTEIN_BUNDLE_ID` / `EINSTEIN_REQUIRE_REAL_BUNDLE` 三个 env var 驱动 disk-backed Wave K artefact 加载，回退路径走 `synthetic_einstein_corpus()`；原 `einstein` vertical 名保留为 backward-compat alias 指向 `einstein-bundle`；[`start_browser_chat_qwen.sh`](../start_browser_chat_qwen.sh) + [`.ps1`](../start_browser_chat_qwen.ps1) 头部 doc + env var 默认值落档。Demo 现场可见差异（**今天就真实存在**）：`einstein-raw` ↔ `einstein-bundle` 的 L4 拒答 / L3 引证 pointer / L1 风格 — 用户在 chat UI vertical dropdown 切换零前端改动。**两条主动延后的 follow-up 已写为 debt**：(a) **#76** 进程级 `default_persona_lora_pool()` 在 3 vertical 同进程共存时无法 per-vertical 隔离 — 当前 debt #40 / #41 守护期下不浮现（synthetic LoRA delta byte-equivalent），#41 真 PEFT-on-Qwen land 后必须给 `LifeformLLMResponseSynthesizer` 加 `persona_lora_enabled: bool` flag；(b) **#77** 离线 3-condition 对照报告生成器（markdown / HTML 静态产物，给法务尽调 / [`docs/moving forward/figure-evidence-packet.md`](moving%20forward/figure-evidence-packet.md) #58 / #59 reviewer 工艺用），本次 BOSS 决策为 `a_chat_only` 主动延后，~0.5 工程天可补。1302 + 54 = 1356 contract / smoke 测试全绿（排除 2 个预存上游 `volvence_zero.owner_hydration` 子包未声明的 import-boundary 失败，与本改动无关）。

> 2026-05-14 update (debt #65 day-counter 整段 deprecated): 节奏分层（"用户处于第几阶段"）已由 `BehaviorProtocol.TemporalArc.progression_signals`（PE-driven 关系阶段）承接，calendar-day routing 是伪需求。配套清理已 land：删 `docs/specs/growth-advisor-day-counter.md` + `tests/contracts/test_growth_advisor_day_routing.py`；`cheng_laoshi.py` 17 处 `applicability_scope` 中 `growth_advisor:dayN` 字符串残留全部清理（funnel/rapport rules 直接删 day 标签；7 个 playbook-dayN rules 改为 onboarding-arc 关系阶段 reserve rule，`description` 从 "Day N" 改为 "Phase: ..."）；7 个 `scenarios/dayN_*.json` rename 为 `sNN_*.json` 同步更新 `scenario_id` + test 期望；`fixture_uptake.py` / `profile.py` / `__init__.py` / `prd.md` / `archetecture.md` / `SYSTEM_DESIGN.md` / `SYSTEM_GUIDE.md` / `DATA_CONTRACT.md` / `protocol-runtime.md` / `external-validation-protocol.md` / `commercialization-assessment.md` / `cross-cutting-foundation-packet.md` / `growth-advisor-pilot-packet.md`（v0.2，G-B 整段下线）/ `commercialization-evidence-rollout.md`（v0.3，月报字段 `day_cohort_activity` → `protocol_phase_cohort_activity`）/ `outbound_scheduler.py` / `realistic_load_growth_advisor.py` / `rollback_drill_growth_advisor.sh` 全部 prose / 注释同步修订。**P2 子包数量从 6 降至 5**（G-A / G-C / G-D / G-E / G-F；G-B 永久下线）。**P2 阻塞条目从 9 降至 8**：原"指责 3 day-counter routing 完全 noop"作废（节奏由 protocol 承接，不是工程缺位）。

> 2026-05-13 update (Companion Bench smoke real-run findings — 5 new debts #71-#75): 跑通 [`scripts/companion_bench/run_companion_bench_smoke.py`](../scripts/companion_bench/run_companion_bench_smoke.py) Qwen + VZ-synthetic F1 family (4 scenarios) 后，从 SMOKE_REPORT v2 + judge robustness replay 导出 5 条 hard-evidence debts。**核心 finding**：(a) [`#71`] Qwen-内 weak proxy judge 6 axis 全 σ > 8（实测平均 23.2），qwen3-max vs qwen-flash 给同一 transcript 差 35 分 → [`#48`] 真 cross-family sweep **不是 nice-to-have，是 leaderboard 准入硬前提**；(b) [`#72`] smoke 默认 `--substrate-mode synthetic` 让 VZ 跑 deterministic echo，VZ 13.79 vs Qwen 74.56 是 substrate 差不是 architectural argument，触发 P5 kill criteria 误报警的根因；(c) [`#73`] SUT subprocess 单 HTTP 400/timeout 整个 SUT 0 bundle (Qwen v2 重跑 25 min timeout 实例) → 缺 retry + arc-isolation；(d) [`#74`] `SubmissionManifest.system_prompt` / `generation_config` 在 `arc_runner` 未注入 SUT → "manifest 与跑分一致"承诺破；(e) [`#75`] `CostTracker.record_perturn_judge` / `record_arc_judge` 在 `run_submission` 未调用 → 即使加 Qwen 价格表 judge cost 仍 None。**Pipeline 状态**：scaffold + wiring 修补就位 (cost.py 加 Qwen 价格 / score_reference 加 subprocess timeout / build_site recursive glob / OpenRouter scaffold 待用户加 key)；评估 evidence 受 #71 #72 阻塞，绝对数字不可外引。

> 2026-05-13 update (rollout v0.2 — 26 条 commercialization debt 全部进入 SHADOW 阶段): 按 [`docs/moving forward/commercialization-evidence-rollout.md`](moving%20forward/commercialization-evidence-rollout.md) §3 W1-W2 + §10 W1 PR 拆法 + §3 推荐起跑顺序，一次性 land 全 4 个 packet 的 SHADOW scaffold（约 62 新文件 + 10 修改）。**本批 SHADOW 不破现有 1452+ contract test**：所有新字段都是 `Optional` 默认（`compatible_substrates=()` / `tenant_identity=None` / `refusal_eval_report=None` / `validated_substrates=()` / `cost_breakdown={}`），所有新 perf test 走 `@pytest.mark.perf` 默认 skip。**26 条 debt 状态**：#45-#70 全部从"未启动"推进到"SHADOW scaffold land + 待 evidence run"——具体每条 debt 对应的 scaffold 文件清单见 rollout v0.2 附录 D。**ACTIVE 推进路径**：W2 起团队按 [`commercialization-evidence-rollout.md`](moving%20forward/commercialization-evidence-rollout.md) §6 周交付物 checkbox 跑（reviewer 招募 → API sweep → GPU PEFT → 30 天试点），不需要再写骨架代码。**关键 SHADOW 选择记录**：(a) F-B 双层 scope `bind_session_two_layer` 是 opt-in（不破 closed-alpha legacy 路径）；(b) F-C `compatible_substrates` 折入 `compute_bundle_integrity_hash` 仅当非空（旧 bundle hash byte-stable）；(c) P5 `ScenarioSpec.language` **不**进 `to_canonical()`（避免 24 公开 scenario hash 全表 rotation）；(d) P2 archetype 识别走 (a) `LLMArchetypeClassifier`（DeepSeek V4 默认）+ (b) keyword 路径被 AST 守门永久排除 + (c) metacontroller 长期过渡。

> 2026-05-13 update (commercialization-assessment §1.1/§4/§6/§8 反向回查): 把 [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) 三个最 promising 方向（P5 Companion Bench / P1 Figure-as-a-Service / P2 Growth-Advisor）的"商业承诺已写进定价/SLO/kill criteria，但仓库里没有对应技术 evidence"的缺口反向回查到工程层，列入 26 条新债 #45-#70：横切 7 条 (#45-#51) + P5 6 条 (#52-#57) + P1 6 条 (#58-#63) + P2 7 条 (#64-#70)。**核心反思**：现有 closed-alpha-api-service / figure-vertical / companion-bench 的 spec 与 contract test 等价于「单实例正确性」evidence；commercialization 文档隐含假设「在生产并发 / 多租户 / 可删除 / 跨 substrate 升级 / 真实用户感受」条件下也有等价 evidence，但这一组数据实际不存在。和 [`docs/moving forward/summary.md`](moving%20forward/summary.md) §2 警告过的"原理免疫 vs 工程已实现混淆"在商业层的同构重现。**P0 6 条**（中-高，Phase A 必做）：#45 生产并发 stress / #48 LLM-judge bias / #52 6 轴权重 calibration / #58 L4 false-refuse-answer GT / #59 L3 引证 faithfulness / #64 boundary 触发率 baseline。**P1 6 条**（中，Phase A 后期）：#46 / #47 / #49 / #63 / #66 / #69。**P2 8 条**（低-中）：#50 / #51 / #55 / #57 / #65 / #67 / #68 / #70。**P3 6 条**（低，持续）：#53 / #54 / #56 / #60 / #61 / #62。最关键单一动作：在 Phase A 第 1-2 个月先建 `tests/perf/` + `tests/multi_tenant/` + `scripts/realistic_load_*.py` 实测床（铺横切 #45 #46 #49 三条），所有报价 / SLO / 合规话术才有底气。

> 2026-05-12 update (Figure-Vertical 端到端验证管线 land + 真跑诊断 — Wave N-P + Wave Q dry-run; 4 道 gate 真跑数据已记录到 debt #39-#42)

> 2026-05-12 update (Figure-Vertical persona verification dry-run — Wave Q): 把 Wave N-P 装好的管线在真 Wave K curated bundle (`figure-bundle:einstein:29eacd226a7cdfd0`, 444 chunks) + tiny-gpt2 substrate + 真 Wave N curated synthetic LoRA (`persona-lora:einstein:61c93126a0b2e98c`) 上跑了一次完整 verification（5 道 in-corpus + 5 道 out-of-scope，artifact 落 `artifacts/figure_verify/einstein-tinygpt2-curated/`）。**结果**：4/4 gates 全 FAIL，退出码 2。这是诚实的输出 —— 验证管线在配置不到位时就该报告 fail，不是 bug。**真跑暴露的 4 个具体债务（追踪在 #39-#42）**：(a) `default_persona_lora_pool()` 是 process-local，bake CLI 进程退出后 verify 进程看不到 → 已就地修了 `_ensure_pool_has_bundle_lora` 在 verify 入口自动从 `bundle.lora` 派生 register（不破 R8 wheel boundary，只读公开 `FigureLoRAArtifact` 字段）；(b) `bundle.coverage_map` 在 Wave K curated bundle 上**过严**：transcript 显示 in-corpus 关于 "relativity / postulate / theory" 的题被 L4 ScopeRefuser 当 OOS 短路 —— 真 production bug，记 #39；(c) synthetic LoRA backend 的 hash-derived 常数 delta 经过 tiny-gpt2 LayerNorm 被吃掉 → bundle ≡ bundle_lora（Wave D `_aggressive_persona_layers` 已知；要 voice gate 量化通过得用真 PEFT-trained LoRA on Qwen），记 #40；(d) tiny-gpt2 ~10M params 不能 model Einstein → cognition_score ≡ 0、L3 evidence ≡ 0；要 cognition / evidence gate 通过必须真 Qwen-1.5B + 真 PEFT bake（这台 Win/CPU 跑不动），记 #41；(e) reviewer-curated 5 道 OOS 探针只 60% 触发 L4 拒答（差 1 道 = 1/5 = 20%），threshold 80% 在小样本上离散度过粗，记 #42。**Wave Q 不是新代码 wave**，是真跑 + 诊断 + 把诊断写回 debt ledger 的运维步骤。verify CLI 修补已落库（`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/cli.py:_ensure_pool_has_bundle_lora`），smoke test 6/6 全绿。

> 2026-05-12 update (Figure-Vertical 端到端口气+认知验证管线 Wave N-P): 把 [`docs/specs/figure-persona-verification.md`](specs/figure-persona-verification.md) 从想法推进到 CI-runnable 自动化验证管线。**Wave N** `bake-lora --corpus-mode curated`：`cmd_bake_lora` 加 curated 分支，`load_curated_corpus_from_cleaning_store` 从 Wave K cleaning store + curator metadata 派生真 envelope 集合 → LoRA 训练数据与 curated bundle 的 domain_package 严格一致；`scripts/figure_bake_einstein_persona_lora.sh` 一键调用（CLI 默认 tiny-gpt2，real run 用 Qwen2.5-1.5B-Instruct）。**Wave O** `lifeform_domain_figure.verification.persona` 子包：(a) `generate_in_corpus_questions` 按 chunk_id 排序遍历 `bundle.retrieval_index.chunk_records` 用固定模板派生 ≤ 20 道立场题 + reviewer-curated `OUT_OF_SCOPE_REFUSAL_QUESTIONS` 5 道（tiramisu / sourdough / Python / car / pop song）；(b) `with_condition` 三条 context manager — RAW (no bundle) / BUNDLE (bundle but pool 临时摘 LoRA) / BUNDLE_LORA (默认 activate)；(c) `run_ablation` 跑 conditions × questions 全网格；(d) deterministic 三族 score：voice (`top80 overlap` × 0.6 + `sentence-length p50 match` × 0.4) / cognition (retrieval_index `assertion_is_supported` 在 GT chunk 上的 cosine) / refusal (`l4_scope_refusal` tag + reviewer-written preamble)；(e) 4-gate verdict — `gate_cognition_improves` (Δ ≥ 0.05) / `gate_voice_improves_with_lora` (Δ ≥ 0.02，载荷性 gate) / `gate_refusal_works` (≥ 0.80) / `gate_evidence_emerges` (≥ 1)，全 deterministic 不依赖 LLM judge。**Wave P** CLI driver `python -m lifeform_domain_figure.verification.persona.cli` 输出 `artifacts/figure_verify/<run_id>/{questions.jsonl, results/<cond>.jsonl, scores.json, verdict.json, transcript.md}`；`scripts/figure_verify_einstein_persona.sh` 端到端编排（默认顺带跑 bake-lora，可 SKIP_BAKE=1）；6 case synthetic-substrate smoke 测试 + `@pytest.mark.hf` 真 Transformers runtime 测试（CI skip）；新 `docs/specs/figure-persona-verification.md` + `figure-vertical.md` mermaid 同步。**Follow-up debts**：(a) voice score 现用 hashing-embedding overlap，未来希望升级到 substrate residual cosine —— 真在 forward 上抽 voice 信号才比 lexical proxy 准；(b) 4 个 gate 阈值是 Wave P 经验值（`cognition_delta=0.05`、`voice_delta=0.02`、`refusal_min=0.80`、`evidence_min=1`），等积累足够多 verdict run 后用 ROC 校准而不是 hand-tune。

> 2026-05-12 update (Figure-Vertical 真材料管线 piloted Wave H-M): 把 [`docs/specs/figure-vertical.md`](specs/figure-vertical.md) 的 "全链真接通" 推进到 "真材料采集 piloted"。**Wave H** `parse_by_content_type` 注册 `text/x-wiki`（与 L0 wikisource `?action=raw` 路径对齐）+ AST contract test 守门 L0/L1 content_type 常量永远对齐；**Wave I** `figure_verify run-batch` 真调度全部 7 个 verifier（前 3 个 first-batch + 4 个 metadata-driven），`--metadata-mode offline/live` 切换 + `--figure-context-file` 一次性传 figure 级常量；缺字段 → `NEEDS_REVIEW`（永不 `missing-check`）；singleton anchors 拿 trivially-PASS 的 cross_source_byte 行；AST contract test 静态守门 `IMPLEMENTED_CHECK_KINDS` 与 CLI 调用对齐；**Wave J** 新 `corpus.loaders.load_curated_corpus_from_cleaning_store` + CLI `bake-bundle --corpus-mode curated --cleaning-root --curated-metadata-file --verification-root --require-verification-pass`，curator 从 L1 cleaning store 一键编出 verified bundle；R15 round-trip 字节稳定性契约测试通过；**Wave K** 真 Einstein 语料采集 piloted：[`packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.jsonl`](../packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.jsonl) 10 个 reviewer-staged URL → 6 SUCCESS / 4 FAILED_HTTP（fail-loud, no silent passes）→ 5 cleaned 文件 → reviewer 选 2 篇 substantive (Einstein 1916 GR 76967 字 + Gutenberg 30155 Relativity 184682 字) → 真 bundle `figure-bundle:einstein:29eacd226a7cdfd0`, `provenance_fingerprint=c156321de6...`；L2 ledger 14 条记录（2 anchors × 7 axes）3 PASS + 4 NEEDS_REVIEW per anchor；**Wave L** 6 类 robustness integration tests + 3 个 `@pytest.mark.live_network` 默认 skip（CI 不阻塞，本地 `pytest -m live_network` 验证 SUCCESS / ETag idempotency / SSRF live 全过 in 74s）；**Wave M** 本条同步。**仍开放**：debt #19 残余（reviewer human-in-the-loop UI + 大批量 curated payload 数据集）/ #27 lu_xun corpus / #28 残余（reviewer workflow / 双盲审）。

> 2026-05-12 update (Figure-Vertical 全链路真接通 Wave A-G): 把 [`docs/specs/figure-vertical.md`](specs/figure-vertical.md) 描述的 L1 / L3 / L4 enforcement chain 从「class 在那里，运行时不调」推进到「synthesize() 真调用」。**Wave A** D2 三件 helper（`compute_dedup_report` / `fingerprint_provenance` / `parse_locator`）真接进 `build_figure_artifact_bundle` 主管线 + GroundedDecoder 暴露 typed `EvidencePointer`（debt #24 closure）；**Wave B** `PEFTLoRABakeBackend.bake` 真训练循环（peft + transformers + torch optional dep；CPU 短 epoch ~5s 跑通；artifact shape 与 synthetic backend 兼容）（debt #18 closure）；**Wave C** `OpenWeightResidualRuntime.capture_for_contrastive` + `build_steering_training_plan(substrate_runtime=...)` 真 hidden-state 抽方向，hashing 路径退役为 fallback（debt #21 closure）；**Wave D** 新 `LoRAAwareResidualRuntime` Protocol + `TransformersOpenWeightResidualRuntime.activate_lora` 真 forward-hook + `PersonaLoRAPool.activate(figure_id, runtime=runtime)` context-manager 真改 forward + 退出字节级回滚 + frozen base `state_dict_hash` 全程不变（R2 + R15 守门）（debt #20 closure）；**Wave E** `_handle_adopt` 主路径自动 `lookup_figure_bundle` + `register_bundle_persona_lora` + `manager.bind_figure_bundle` + `Lifeform.bind_figure_bundle` 透传到每个 session 的 synthesizer（debt #22 closure）；**Wave F** `LifeformLLMResponseSynthesizer.synthesize` 真嵌入 ScopeRefuser pre-check（STRICT_REFUSE 短路）+ StylePriorInjector hint tag + `pool.activate(figure_id, runtime=runtime)` context wrapper + GroundedDecoder post-verify tag；**Wave G** `test_full_chain_e2e_real_wiring.py` 一次性把 corpus → real residual steering → real PEFT LoRA → OFFLINE gate → pool register → activate over Transformers runtime → logit shift → byte-identical deactivate → enforcer wired 全链跑通。507 tests 全绿（含 `@pytest.mark.hf` 真 HF stack）。**仍开放**：debt #19（curated 真 corpus 数据集）/ #27（lu_xun corpus）/ #28（残余 reviewer 工艺）— 都是「真材料」类工作，本轮按用户范围排除。

> 2026-05-11 update (Companion Bench v1.0 public launch — rename Companion Bench → Companion Bench + full site landed): #29 轨 2 跨过 reference-impl → public-launch 边界。**Code rename**：`packages/companion-bench/` → `packages/companion-bench/`，Python module `companion_bench` → `companion_bench`，CLI `companion-bench` → `companion-bench`，所有 4 个 GitHub workflow rename 为 `companion-bench-*.yml`，`scripts/companion_bench/` → `scripts/companion_bench/`，contract test rename + 内部断言更新；145 个 unit + contract + pipeline test 全绿；24 个 SHA-256 scenario hash 经验证 byte-identical（bench name 不在 `ScenarioSpec.to_canonical()` 里，所以 RFC §3 P3 reproducibility 契约自动延续）。**Doc rebrand**：8 个 `docs/external/companionbench-*.md` + `docs/specs/companion-bench.md` rename + 内部 Companion Bench → Companion Bench 文案统一，旧路径留 5-line redirect stub（一发 release 后删）；`docs/specs/companion-bench.md` + RFC + governance / submission / crosswalk / heldout-bootstrap / hash-manifest 全部 sync。**Eqbench-parity site**：`site/leaderboard/` 单页 → `site/` 9 个 page（landing + leaderboard + methodology + scenarios + submit + governance + judges + compare + about）+ `site/results/?s=<id>` 单模板 detail page（per-axis bars + per-scenario transcripts + callback ledger + per-turn rubric heatmap + cost）+ `site/compare.html` pairwise side-by-side viewer（synced turn highlight + per-axis margin bars）+ `site/judges.html` quarterly rotation / Spearman agreement / calibration scatter；cozy light + dark theme（warm ivory `#fffbf8` / dark `#1f1b18`）+ `assets/theme.js` 持久 localStorage；inline-SVG charts (`assets/charts.js`：bar + forest + heatmap + scatter)，零外部 chart 库依赖。**build_site.py**：把 `artifact_dir/<submission_id>/{summary.json,*.bundle.json}` 编译成 `site/data/aggregate_results.json` + `site/data/submissions/<id>.json` + `site/data/pairwise.json`（TrueSkill + BT + per-arc winners） + `site/data/scenarios.json`；端到端测试 [`packages/companion-bench/tests/test_build_site_pipeline.py`](../packages/companion-bench/tests/test_build_site_pipeline.py) 用 deterministic-fake 跑两个 submission 验通。**Demo data**：[`scripts/companion_bench/populate_demo_site.py`](../scripts/companion_bench/populate_demo_site.py) 在没有真 API key 时用 deterministic-fake 跑 8 个 mock submission × 24 scenario × 1 seed = 192 arc 充满 site/data，标 `demo: true` 显 banner；真 reference 跑分用 `score_reference_systems.py`（unchanged orchestrator + new build_site 出口）。**Launch hardening**：`site/CNAME` → `companion-bench.org`、`site/robots.txt`、`site/sitemap.xml`、SVG favicon + og-image、`.github/ISSUE_TEMPLATE/{submission-request,bug-report,config}.yml`、citation/BibTeX in landing + about、Pages workflow `companion-bench-publish.yml` 改 `path: site` 部署整站。**仍未做（组织 / 预算 / 时机层 follow-up）**：(a) 真 10 reference systems 跑分（$500-3500 API 预算）— 入口已就位，等批准；(b) DNS / GitHub Pages CNAME 真生效（registry register `companion-bench.org` + DNS A record + Pages settings custom-domain 三步，~30 分钟）；(c) `companionbench/heldout` private repo 创建 + deploy key（heldout_loader 已支持 legacy `external/companionbench-heldout/` alias）；(d) working group 形成 / RFC v0.1 → v1.0 升级（~6 周公开评论期）；(e) 一发 release 后清掉 7 个 Companion Bench redirect stub。这五项均不阻塞 v1.0 站点上线，但都追踪在 #32（Companion Bench v1.0 launch path）。

> 2026-05-11 update (EQ-Bench 3 wiring dry-run validated + Companion Bench v1.0 reference impl land + Real-Person Figure Vertical F1-F6 + Persona Figure 数据管线 V1 D2/D3/D4/D7 全套 land；#18-#22 为 figure F6 训练后端/数据/DLaaS hook，~~#23~~ 已闭合（figure bake CLI + bundle 持久化 + audit log + rollback 全套 land），~~#19~~ 部分闭合（V2 fetcher + parser 落地）、~~#25~~ 闭合（metadata fingerprint folded into bundle hash）、~~#26~~ 闭合（4 V2 metadata clients + cache + role gate）；#28 完整 webcrawl 编排 + 清洗管线 + 多源验证审计三层（L0+L1+L2 first batch+L2 second batch）**架构层完全 land**，剩余只是 curated 数据集 / reviewer workflow 类工作；#24/#27 为本轮 D2/D7 数据管线层未串接缺口；**#29-#30 为对外 benchmark 证据面缺口（融资尽调阻塞条件）**，#29 P1-P9 已 land + **wiring 层已通过 dry-run 验证**（见 [`docs/external/eqbench3-wiring-evidence.md`](external/eqbench3-wiring-evidence.md)：synthetic vertical 上 45/45 scenarios 跑通 + 修复 2 个 URL bug），**P10 actuation（真 Qwen substrate + 真 Anthropic key + 三轨 ablation + verdict）独立追踪在 #37**；**#31 OpenAI-compat 适配 wheel 流式 SSE 未实现**；**#29 轨 2 Companion Bench v1.0 reference impl 已 land**：`packages/companion-bench/` Apache 2.0 wheel + 24 public + 96 held-out scenarios + 10 reference systems orchestrator + GitHub Pages 站点 + 4 个 CI workflow；Companion Bench v1.0 launch 路径未启动追踪在 **#32**（governance / 真 reference 跑分 / DNS / submission queue），human-eval 轨道 0% 追踪在 **#33**（RFC §6.6），harness 性能 / async / staged executor 追踪在 **#34**，季度 rotation 自动化（held-out / lexicon / judge）追踪在 **#35**，v2.x 长尾（multi-modal / EQ-Bench 1:1 prompt / 加密 attestation / transcript 脱敏）追踪在 **#36**——配套对外 RFC 见 [`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md)，配套 OpenAI-compat 适配层位于 `packages/lifeform-openai-compat/`）

> 2026-05-11 update (EQ-Bench 3 wiring dry-run validated): 把 #29 P1-P9 已交付的代码层从「unit tests + 文档」推进到「真 harness 端到端跑通」。Clone [`https://github.com/EQ-bench/eqbench3`](https://github.com/EQ-bench/eqbench3) 到 `external/eqbench3/`、装 9 个 light deps、boot `lifeform-serve --vertical companion --substrate-mode synthetic --enable-openai-compat`、配 `.env`、跑 `eqbench3.py --no-rubric --no-elo --ignore-canonical` 全 45 scenarios → 全部 `status: completed`、产 [`artifacts/external_bench/eqbench3_dry_run.runs.json`](../artifacts/external_bench/eqbench3_dry_run.runs.json) (1.17 MB)、26 debriefs 也全通过。**修复 2 个 URL bug**：(a) [`scripts/external_bench/.env.example`](../scripts/external_bench/.env.example) + [`run_eqbench3.py`](../scripts/external_bench/run_eqbench3.py) `TEST_API_URL` / `JUDGE_API_URL` 原本只写 `/v1` 但 eqbench3 `utils/api.py` 不会补 `/chat/completions` → 改为 full path，11 个 `test_run_eqbench3_smoke.py` 单元测试在 fix 后全绿；(b) PowerShell `Out-File -Encoding utf8` 默认带 BOM 破坏 dotenv 解析 → bootstrap note 写入 [`docs/external/eqbench3-wiring-evidence.md`](external/eqbench3-wiring-evidence.md)。Wallclock：synthetic CPU 单 track 14:42。**P10 actuation 仍 gate 在**：真 Qwen 1.5B substrate（GPU 或 hf-shared）+ 真 Anthropic `JUDGE_API_KEY` + 调用 `--with-elo` 跑 pairwise（增量 ~$10-20/track）+ 三轨 (`companion / companion-cold / raw`) 跑完产 `.summary.json` 后 [`compare_ablation.py`](../scripts/external_bench/compare_ablation.py) 取 verdict。完整 reproduction recipe 见 evidence 文档。这一步**把「实测前的所有未知」从 #29 的债务面剥离** — 接下来真跑分时只剩 substrate / judge 一类 known-unknown，adapter-side 隐藏 bug 已清零。

> 2026-05-10 update (Companion Bench v1.0 reference impl land — debt #29 轨 2 推进): RFC 文档级 v0.1 → 工程级 v1.0 全套就位。新 wheel [`packages/companion-bench/`](../packages/companion-bench/) Apache 2.0 隔离许可，13 模块 + 144 单元测试全绿（包括 `tests/contracts/test_companion_bench_no_internal_imports.py` 静态守 companion-bench 不 import vz-* / lifeform-*，匹配 RFC §3 P4 outcome-level 评估契约）。**Held-out 治理**：96 scenario 走 git submodule + private repo `companionbench/heldout`（[`.gitmodules`](../.gitmodules) + [`docs/external/companion-bench-heldout-bootstrap.md`](external/companion-bench-heldout-bootstrap.md) 一次性 organiser bootstrap），公仓 PR 永不见 held-out body；CI release-tier 用 deploy key 拉取，PR / open-source clones 自动跳过。**Public scenarios**：24 个完全 in-repo（6 family × 4），hash 表落 [`docs/external/companion-bench-public-scenario-hashes.txt`](external/companion-bench-public-scenario-hashes.txt) 由 [`scripts/companion_bench/emit_scenario_hashes.py`](../scripts/companion_bench/emit_scenario_hashes.py) 重生成。**Public leaderboard 静态站**：`site/leaderboard/` 纯 HTML + vanilla JS，[`scripts/companion_bench/generate_demo_aggregate.py`](../scripts/companion_bench/generate_demo_aggregate.py) 给出 demo 渲染数据（10 系统 placeholder），真 reference 跑分入口 [`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 跑通即替换。**4 个 CI workflow**：`companionbench-ci-smoke`（PR gate, 公开） / `companionbench-paper-suite-small`（nightly $200-400） / `companionbench-paper-suite-full`（release $5-15k, 拉 held-out） / `companionbench-leaderboard-publish`（GitHub Pages）。**4 个 shell 脚本**：`run_companion_bench_ci_smoke.sh` / `run_companion_bench_paper_suite_small.sh` / `run_companion_bench_paper_suite_full.sh` / `build_leaderboard_site.sh`。**5 个 governance 文档**：[`companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md) / [`companion-bench-governance-charter-draft.md`](external/companion-bench-governance-charter-draft.md) / [`companion-bench-eqbench-crosswalk.md`](external/companion-bench-eqbench-crosswalk.md) / [`companion-bench-heldout-bootstrap.md`](external/companion-bench-heldout-bootstrap.md) / [`docs/specs/companion-bench.md`](specs/companion-bench.md)。**仍未做（组织层 / 预算层 follow-up，不是代码层债）**：(a) working group 形成（RFC §11，依赖外部第二个组织接入，charter draft 已就位）；(b) 真 10 reference systems 跑分（$5-15k API 预算，scripts 已就位等批准）；(c) 真域名 `companion-bench.org` DNS + GitHub Pages CNAME 配置；(d) v1.1 quarterly held-out paraphrase rotation。这四项都不阻塞 v1.0 reference impl 的工程交付。

> 2026-05-10 update (debt #29 P1-P9 land + new debt #31 streaming SSE): 对外 EQ benchmark 提交全链路（debt #29 推荐修法 1-5）的代码层 + 文档层全部就位。新 wheel `lifeform-openai-compat` 暴露 `POST /v1/chat/completions`（OpenAI envelope）on top of `lifeform-service`，三轨 ablation 设计（companion / companion-cold / raw substrate）支持 EQ-Bench 3 一次跑出"系统 vs 裸 substrate"delta；`scripts/external_bench/run_eqbench3.py` + `compare_ablation.py` 提供 launcher + verdict 守门（红线 attestation 缺失则拒绝出 verdict）；`scripts/external_bench/run_empathybench.py` 提供同结构 generic harness（empathybench.com 闭源时支持 EmotionBench 等开源等价物）。文档层落 [`docs/external/eqbench3-submission-protocol.md`](external/eqbench3-submission-protocol.md)（reproducibility 协议）+ [`docs/external/companion-bench-eqbench-crosswalk.md`](external/companion-bench-eqbench-crosswalk.md)（Companion Bench↔EQ-Bench 跨 benchmark 映射，给 Companion Bench v0.2 evidence backing）+ [`docs/external/eqbench3-results-internal.md`](external/eqbench3-results-internal.md)（per-run verdict 模板）+ [`docs/external/eqbench3-public-submission-checklist.md`](external/eqbench3-public-submission-checklist.md)（P10 actuation gate）。**1048 import-boundary tests + 138 adapter unit/integration tests 全绿**，`vz-* / lifeform-* / lifeform-domain-*` 内核包**零修改**，唯一一处 lifeform-service 改动是 `cli.py` 加 `--enable-openai-compat` flag（默认 off，向后兼容）。守红线静态化：[`tests/contracts/test_openai_adapter_import_boundary.py`](../tests/contracts/test_openai_adapter_import_boundary.py) AST 守 adapter 不导内核；[`tests/contracts/test_openai_adapter_no_kernel_writeback.py`](../tests/contracts/test_openai_adapter_no_kernel_writeback.py) AST 守 adapter 不写 SessionManager / LifeformSession 私有状态；`compare_ablation.py` 程序化校验四条 #29 红线（frozen substrate / no kernel mod / no benchmark text in system prompt / no internal vocab in model card）后才发 verdict。**P10 公开提交仍 gate 在 verdict==go**，需要先用真实 GPU + Anthropic API key 跑一次 ablation 拿到分数 — 这一步不在代码 packet 范围。**新增 debt #31**：adapter 当前 `stream=true` 返 501，对 EQ-Bench 3 这种 single-shot harness 不影响，但任何 streaming 模式 harness（Chatbot Arena 实时投票通道、OpenAI Python SDK streaming chat）会因此被阻塞 — 与 #30 (Chatbot Arena 提交) 同时浮现，需要 SSE 落地。
>
> 2026-05-10 update (Companion Bench RFC + 对外 benchmark 证据债 #29-#30): 对外 benchmark / arena 评估面（业界正在快速成形的 EQ-Bench 3 / EmpathyBench / RP-Bench / Chatbot Arena 等）我们当前**完全缺位**——这条缺位本身既不影响系统运行也不违反 R 铁律，但对 **任何 fundraising 尽调和品类话语权竞争** 都是硬阻塞。本轮把 follow-up 拆成三轨：(轨 2) 已落地：[`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md) 公布 Companion Bench (Long-Session Companion Benchmark) 公共 RFC v0.1，从中立角度定义 multi-session companion 评估方法学（A1–A6 六轴 + 六族 scenario），不暴露任何内部架构（NL/ETA/R-PE/regime/owner SSOT/family report 内部口径），目的是让公司在长会话陪伴评估这条赛道上**先于打榜成为 convener**；与 EQ-Bench 3 rubric 兼容以便他人系统已有信号可以转移；(轨 1) 列入 #29：把 substrate / lifeform 包一层 OpenAI-compatible API 提交 EQ-Bench 3 + EmpathyBench，拿一个**可被引用的客观分数**填上"投资人尽调时第一个 google 到的数字"这条空白；(轨 3) 列入 #30：在人评类 arena（RP-Bench / Chatbot Arena 公开 chat / 后续 Companion Bench 自研人评轨）建立可见 footprint，对齐"EQ > IQ"这条对外叙事。轨 2 不在 known-debts；轨 1/3 短期可推进、与代码层债项独立，写为 #29/#30 跟踪。
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

## 12. DLaaS Slice 5.4 真流式 SSE（API 层 2026-06-11 已落地；substrate token hook 仍欠）

> 2026-06-11 update（API 层 SSE 落地，debt 部分关闭）：新增
> [`packages/dlaas-platform-api/src/dlaas_platform_api/streaming.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/streaming.py)，
> `_dispatch_envelope_to_instance`（本地 + multi-pod forward 两条路径）在
> `output_contract.stream=true` 且 interaction_type ∈ {chat, teach, task} 时按
> `DLAAS_README.md` 文档化的 `event: ack → chunk* → act* → done`（失败为终止性
> `event: error`，绝不 silent EOF）回 SSE；`ack` 在 kernel turn 之前写出，`done`
> 携带与 JSON 路径完全同形的 body。observe/feedback/report/command 与 paused
> takeover 路径按 `OutputContract` best-effort 条款静默降级为 JSON。审计 / usage /
> cognition snapshot 记账与 JSON 路径逐一对齐（audit payload 多带 `stream: true`）。
> 测试：`packages/dlaas-platform-api/tests/test_interaction_streaming.py`（7 个）。
> spec 同步：[`docs/specs/dlaas-api-v1.md`](specs/dlaas-api-v1.md) §"SSE response"。
> **仍欠（本 debt 保持 open 的部分）**：chunk 目前是 kernel 整段产出后的平台层切
> 段（presentational），不是真 token 增量——wire contract 已就位，token 粒度等
> substrate streaming additive 接口（下方修法 1-2）单独 review 落地后自动升级。

- **路径**：
  - API 层 SSE（已落地）：[`packages/dlaas-platform-api/src/dlaas_platform_api/streaming.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/streaming.py)
  - dispatch（仍是整段产出）：[`packages/dlaas-platform-api/src/dlaas_platform_api/dispatch.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/dispatch.py)（chat / teach / task 三个 handler 直接 `await session.run_turn(...)` 后整段返回）
  - 缺位的 substrate hook：[`packages/vz-substrate/src/volvence_zero/substrate/`](../packages/vz-substrate/src/volvence_zero/substrate/)（`OpenWeightResidualRuntime.generate(...)` 是 sync block；没有 `generate_async` / `stream_tokens` 接口）
- **问题（剩余部分）**：SSE wire contract 已实现，但 chunk 粒度受限于 kernel 的原子整段生成；真 token-level 增量需要 substrate streaming additive 接口。该项是 DLaaS 切片中**唯一可能动 `vz-substrate` 的位置**，需单独 review。
- **违反**：纯产品 UX 体验差，不违反 R2/R4/R8 任何铁律——cancel 的理由是"动 substrate 需要单独 review"，不是动了会出错。另注意：expression 层 enforcer（GroundedDecoder / ScopeRefuser）作用于完整文本，真 token 流式落地时必须保证 chunk 不绕过这些 gate（在 enforcer 之后流出，或 enforcer 支持增量模式）。
- **风险**：低。first-byte 延迟已由 `ack` + 平台层 chunk 解决大半；剩余是长生成期间 chunk 仍一次性到达的体验差距。**不影响功能正确性**。
- **触发条件**：(a) 第一个真实生产集成提出"必须 token-level 流式"的需求；(b) 某个 vertical 的 chat 平均生成时间稳定 > 5s；(c) 接 LLM judge 后发现 evaluation 端的 token 流也需要流式 readout（关联 #13）
- **推荐修法（剩余部分）**：
  1. `vz-substrate` 加 additive `async def generate_async(self, prompt, *, on_chunk: Callable[[str], None]) -> str` 接口（不删 / 不改现有 `generate(...)`，新方法独立测试套）
  2. `lifeform-expression.LifeformLLMResponseSynthesizer` 加 `synthesize_streaming(...)` 派生方法（注意 enforcer gate 不被绕过）
  3. ~~SSE writer~~ —— 已落地（`streaming.py`），token hook 接入时只改 `run_dispatch` 内部产 chunk 的来源，wire contract 不变
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

## ~~18. Figure F6 PEFT LoRA bake backend 是 stub，真 GPU 训练未接~~ —— 2026-05-12 关闭

> **关闭说明（2026-05-12，Wave B of "接通 Figure-Vertical 全链路（除真材料外）" packet）**：
> - `peft>=0.11` 进 [`packages/vz-runtime/pyproject.toml`](../packages/vz-runtime/pyproject.toml) 的 `[project.optional-dependencies].torch` extra；`pip install vz-runtime[torch]` 一键启用。
> - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_peft.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora_bake_peft.py) `PEFTLoRABakeBackend.bake` 替换 `NotImplementedError`：HuggingFace 短 epoch + AdamW + 每 `target_module` 抽 LoRA A/B 矩阵 → `B @ A` 扁平化 + sign-preserving bucket pool 到 `delta_vector_dim` → `tuple[SubstrateDeltaAdapterLayer, ...]`，与 `SyntheticLoRABakeBackend` 输出 schema 兼容；`validation_delta = (init_loss − final_loss) / init_loss`、`capacity_cost = trainable / total params` 真值进 OFFLINE gate proposal。
> - 新 `PEFTLoRAConfig` typed dataclass：`target_modules` / `rank` / `alpha` / `dropout`；`PEFTLoRABakeBackend` 加 `model_id` / `peft_config` / `runtime_device` / `max_steps` / `checkpoint_dir` / `delta_vector_dim` 字段。
> - CLI [`packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py) 解除 `--backend != synthetic` 的 fail-loud；新增 `--peft-model-id` / `--peft-target-modules` / `--peft-alpha` / `--peft-dropout` / `--peft-max-steps` / `--peft-device`。`scripts/figure_demo_einstein.sh` 加 `BACKEND=peft` 一键切。
> - 测试：`packages/lifeform-domain-figure/tests/test_lora_bake_smoke.py::test_peft_backend_bake_real_loop_smoke`（`@pytest.mark.hf`）+ `tests/contracts/test_figure_persona_lora_synthetic_vs_peft_shape.py` 双 backend shape 兼容契约（pool 同时接受两端）。CPU 短 epoch 在 `sshleifer/tiny-gpt2` 上 < 5s 跑通。
> - 关键不变量：bake 完仍走 `apply_persona_lora_through_gate` 的 OFFLINE gate（R10 不变）；frozen base 在训练 / activation 全程不变（peft 在 deep-copy 模块上跑，原 model.state_dict() 字节稳定）。

## ~~18. (closed)~~

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

## ~~19. Figure vertical 默认 corpus 是 reviewer-paraphrased synthetic，archive V2 fetcher + 真 payload 数据集 + parser 三件未做~~ —— 2026-05-12 大部分关闭

> **进度（2026-05-10）**：本债的 (1) V2 archive fetcher + (2) parser 两件已落地，随 debt #28 L0 + L1 packet 一并实现：
>
> - **V2 archive fetcher**（修法 1）：`corpus.archives.live_archive_fetcher(fetch_kind, ...)` 工厂返回 V2 `ArchiveFetcher`-Protocol 实现；4 个 archive 各有专用 URL pattern + content-type 选择（CPAE PDF / Wikisource action=raw / Gutenberg .txt / IA metadata API → OCR JSON）；既有 `offline_archive_fetcher()` 行为不变（V1 向后兼容）。SSRF allowlist + body-cap + 1-hop redirect rescope 在 `BaseHTTPClient` 集中守。详见 [`docs/specs/figure-corpus-crawl.md`](specs/figure-corpus-crawl.md) 末尾 "V2 ArchiveFetcher closure" 段。
> - **parser 套件**（修法 2）：随 debt #28 L1 落地（CPAE PDF via pypdf / Wikisource HTML via bs4+mwparserfromhell / Gutenberg HTML+plain via bs4 / IA OCR JSON via stdlib）。详见 [`docs/specs/figure-corpus-cleaning.md`](specs/figure-corpus-cleaning.md)。
>
> **进度（2026-05-12 Wave J + Wave K）**：剩余 4 块也已 land：
>
> - **(3) curated payload 数据集 path**：[`packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.jsonl`](../packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.jsonl) 落第一份 reviewer-staged seed list（10 URL × 3 archive kinds）；同目录 `*.curated_metadata.jsonl` 落第一份 `CuratedSourceMetadata` 集合（每条 keyed by `raw_sha256`，archive-specific payload + provenance fields）。新 [`packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/loaders.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/corpus/loaders.py) 的 `load_curated_corpus_from_cleaning_store(cleaning_root=..., figure_id=..., metadata_file=...) -> CuratedCorpusBundle` 把 L1 cleaning store + 元数据 JSONL 编进 `Figure*Source` + `SourceProvenance` 两组数据，喂给 `build_figure_artifact_bundle(...)` 直接产 verified bundle。
> - **(4) `figure-bake bake-bundle` CLI 加 `--corpus-mode curated --cleaning-root --curated-metadata-file --verification-root --require-verification-pass`**：从 reviewer 收集到的真字节流一键编出 verified bundle。
> - **(5) `provenance_fingerprint` 字段**：早在 Wave A (2026-05-12) 加进了 [`FigureArtifactBundle`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/figure_artifact.py)；非空时折入 `integrity_hash`（任何 license / capture_method 漂移都让 bundle id 变）。
> - **(6) 单文档 fetcher → 完整 webcrawl/clean/verify**：debt #28 三层 (L0+L1+L2) 已全 land；Wave H closure 把 wikisource `text/x-wiki` 路径的真字节流 dispatcher 补齐，Wave I closure 把 `figure_verify run-batch` 的 4 个 metadata-driven verifier 接通。
>
> **首次真采集证据**：[`scripts/figure_collect_einstein.sh`](../scripts/figure_collect_einstein.sh) → 6 SUCCESS / 4 FAILED_HTTP / 5 cleaned 文件 / reviewer curate 2 篇 → bundle `figure-bundle:einstein:29eacd226a7cdfd0` (provenance_fingerprint=`c156321de6...`)。L2 ledger 14 行 (3 PASS + 4 NEEDS_REVIEW per anchor)。
>
> **本债真正剩余开放**：(7) **大批量 curated 数据集**（10–20 份 → 100+ 份；reviewer 工艺时间）；(8) **reviewer human-in-the-loop UI**（替代手写 JSONL；产品决策）；(9) **`figure_bundle_store._seed_default_store` 切到 curated-by-default**（产品决策，等 (7) 数据量够大）；(10) **tenant-supplied corpus 路径**（外部 ingestion，不在 figure-vertical 范围）。本节剩余工作不再是 figure-vertical 架构缺位，全部是产品 / 工艺 / 法务侧 follow-up。
>
> _（保留下方原始描述以便 audit；新工作应在 debt #28 / 数据集 follow-up / 产品 reviewer UI follow-up 中追踪。）_

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

## ~~20. PersonaLoRAPool.activate 是 in-memory passthrough，未接真 GPU 多 LoRA 热插~~ —— 2026-05-12 关闭

> **关闭说明（2026-05-12，Wave D of "接通 Figure-Vertical 全链路（除真材料外）" packet）**：
> - 新增 [`packages/vz-substrate/src/volvence_zero/substrate/lora_aware_runtime.py`](../packages/vz-substrate/src/volvence_zero/substrate/lora_aware_runtime.py)：`LoRAAwareResidualRuntime` Protocol（`activate_lora(layers) -> AbstractContextManager[None]`），结构化类型，任何 runtime 暴露此方法即满足。
> - [`packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py`](../packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py) `OpenWeightResidualRuntime.activate_lora` 默认实现：no-op + 进出活动状态记账，禁嵌套（再入抛 `RuntimeError`），不污染 frozen base。`SyntheticOpenWeightResidualRuntime` 继承默认。
> - [`packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py`](../packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py) `TransformersOpenWeightResidualRuntime.activate_lora` 真 forward-hook 覆盖：在 `_block_modules[layer_index]` 上注册 `register_forward_hook`，把 `delta_vector` broadcast 加到该 attention block 的 residual 输出；context 出口移除 hook。Layer-index mismatch（baked 与 hooked 不一致）走 `min(hooked, key=abs(idx - normalised))` 就近映射，加性叠加而非静默丢弃。
> - [`packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py`](../packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py) `PersonaLoRAPool.activate(record_or_figure_id, *, runtime=None)` 改造为 context-manager：`runtime is None` → 维持 legacy passthrough；`runtime` 为 `LoRAAwareResidualRuntime` → 包装 `runtime.activate_lora(record.adapter_layers)` + yield record；`runtime` 不满足 Protocol → 抛 `TypeError` (fail loud)。
> - 测试：`packages/vz-substrate/tests/test_lora_aware_runtime_smoke.py` 7 case：(a) Protocol 结构化类型识别；(b) pool.activate 双签名；(c) 嵌套抛错；(d) `@pytest.mark.hf` 真 Transformers runtime activate 改 logits + 退出字节级回滚；(e) frozen base `state_dict_hash` 在 activate 上下文进出时全程不变（R2 守门）。
> - 与 R2 / R15 关系：activate 改的是 controller 层的 forward hook，base parameters 字节级不变；frozen base 在 activation 内/外/前 三次 hash 全相同；rollback = 退出 context 即可，append-only 跟 #23 持久化无冲突。

## ~~20. (closed)~~

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

## ~~21. F5 Steering bake 在 hashing-embedding 坐标系，未在 substrate 真残差流上提取方向~~ —— 2026-05-12 关闭

> **关闭说明（2026-05-12，Wave C of "接通 Figure-Vertical 全链路（除真材料外）" packet）**：
> - [`packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py`](../packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py) 加 `OpenWeightResidualRuntime.capture_for_contrastive(positive_texts, negative_texts, *, layer_index)` 默认实现：走公开 `capture(source_text)` 抽指定 layer 的 `ResidualActivation.activation`，跨文本均值池化，返回两侧 `tuple[float, ...]` 同维度。
> - [`packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py`](../packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py) `TransformersOpenWeightResidualRuntime` 提供 `_mean_residual_at_layer` 批量 override：复用现有 `_capture_hidden_state_means`（已有的 per-layer hidden-state mean hook），CPU 上 ~2x 快。
> - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_data_prep.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_data_prep.py) `build_steering_training_plan(contrast_set, *, substrate_runtime=None, layer_index=0)`：`substrate_runtime` 提供时走真 hidden-state 抽方向（per-pair `capture_for_contrastive` 一次），`embedding_dim = runtime hidden_size`；`substrate_runtime=None` 时仍是旧 256-dim hashing fallback（SHADOW-safe）。两个路径产出的 `integrity_hash` 不同，下游 OFFLINE gate 可凭此区分坐标系。
> - CLI [`packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py) `cmd_bake_steering` 加 `--use-real-residual` / `--real-residual-model-id` / `--real-residual-layer-index`；audit log `backend_id` 在两条路径下分别为 `"steering-real-residual-v1"` / `"steering-cpu-contrastive-v1"`。
> - 测试：`packages/lifeform-domain-figure/tests/test_steering_bake_real_residual_smoke.py` 6 case，含 `@pytest.mark.hf` 真 Transformers runtime + 假 runtime 双轨；assert positive vs negative 残差均值欧氏距离 > epsilon（不会塌缩为零向量）；hashing 路径 fallback 行为不变。

## ~~21. (closed)~~

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

## ~~22. DLaaS adopt 主路径未自动 hook `lookup_figure_bundle` + `register_bundle_persona_lora`~~ —— 2026-05-12 关闭

> **关闭说明（2026-05-12，Wave E of "接通 Figure-Vertical 全链路（除真材料外）" packet）**：
> - [`packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py`](../packages/dlaas-platform-api/src/dlaas_platform_api/control_plane.py) `_handle_adopt` 在 `instance_manager.acquire(...)` 之后：(a) 读 `template.figure_artifact_id`；(b) 调 `lifeform_service.lookup_figure_bundle(default=None, bundle_id=...)`；(c) 调 `instance_manager.get(ai_id).bind_figure_bundle(bundle)`；(d) 调 `lifeform_service.register_bundle_persona_lora(bundle)`（pool 默认走进程级 `default_persona_lora_pool()`）。Import 是 try/except guarded 所以 platform-only 安装（无 figure wheel）不影响其他 vertical 的 adopt。
> - [`packages/lifeform-service/src/lifeform_service/session_manager.py`](../packages/lifeform-service/src/lifeform_service/session_manager.py) 加 `SessionManager.bind_figure_bundle(bundle)` + `figure_bundle` 属性；`create_session` 在工厂返回 lifeform 后调 `lifeform.bind_figure_bundle(bundle)` 透传。
> - [`packages/lifeform-core/src/lifeform_core/lifeform.py`](../packages/lifeform-core/src/lifeform_core/lifeform.py) 加 `Lifeform.bind_figure_bundle(bundle)` + `figure_bundle` 属性；`_maybe_clone_synthesizer_for_session` 在 clone 同步 `synthesizer.with_figure_bundle(bundle)`，所以 per-session synthesizer 真带 bundle。
> - 测试：`tests/service/test_dlaas_adopt_loads_figure_bundle.py` 5 case：(a) `lookup_figure_bundle("einstein")` 命中默认 store；(b) bundle 无 lora → register 返回 None 且不污染 pool；(c) bundle 有 lora → register 推 adapter_layers 进 pool；(d) `manager.bind_figure_bundle(bundle)` → 后续 session 的 synthesizer.figure_bundle == bundle；(e) `register_bundle_persona_lora(bundle)`（不传 pool）→ 进程级 default pool 命中。

## ~~22. (closed)~~

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

## ~~24. Figure D2 三件 corpus helper（dedupe / provenance / citation parser）未接进 bundle 主管线~~ —— 2026-05-12 关闭

> **关闭说明（2026-05-12，Wave A of "接通 Figure-Vertical 全链路（除真材料外）" packet）**：
> - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/retrieval_index.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/retrieval_index.py) `build_figure_retrieval_index` 加 `chunk_id_allowlist: frozenset[str] | None = None` 参数；`compiler.build_figure_artifact_bundle` 主路径调 `compute_dedup_report(envelopes)` 后把 canonical chunk ids 喂进去，跨 envelope 去重在 BM25 corpus 统计前生效。
> - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/figure_artifact.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/figure_artifact.py) `FigureArtifactBundle` 加 `provenance_fingerprint: str = ""` 字段；`compute_bundle_integrity_hash` 的 payload 在非空时折入 `("provenance", fingerprint)`。同 profile + 不同 license 必产生不同 bundle hash（license-only 漂移在 R15 audit chain 上可见）。
> - [`packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/compiler.py) `attach_steering_to_bundle` / `attach_lora_to_bundle` 把 `provenance_fingerprint` 传穿（rebake 不丢 audit chain）。
> - [`packages/lifeform-expression/src/lifeform_expression/grounded_decoder.py`](../packages/lifeform-expression/src/lifeform_expression/grounded_decoder.py) 加 typed `EvidencePointer`：在 `verify_with_pointers` 返回路径上调 `lifeform_domain_figure.corpus.citation.parse_locator`（lazy import + ImportError 兜底），把 `sender_id` / `recipient_id` / `date_iso` / `venue_id` / `volume` / `page` / `paragraph_index` / `offset` 等结构化字段填到 pointer 上；`rendered` 给出比 raw locator 更可读的引证字符串（如 `letter[einstein->bohr@1935-04-12] | env_id`）。
> - 测试：`tests/contracts/test_figure_bundle_dedup_and_provenance.py` 6 case：(a) 同 paragraph 出现在 paper + letter 两个 envelope → 过滤后只剩 paper 端 canonical 副本；(b) 同 profile + 不同 license_label → bundle hash 不同；(c) 空 provenance → fingerprint 空 + bundle hash 走 legacy 字节稳定路径；(d) GroundedDecoder evidence pointer 真带 parsed 结构化字段；(e) parse 失败 fallback 到 raw locator。

## ~~24. (closed)~~

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
> - 公开文档：[`docs/external/eqbench3-submission-protocol.md`](external/eqbench3-submission-protocol.md) + [`docs/external/companion-bench-eqbench-crosswalk.md`](external/companion-bench-eqbench-crosswalk.md)
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
  - Companion Bench RFC（轨 2，[`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md)）正在做"**定义新评估范式**"的长期防御；轨 1 / 轨 3 是补"**短期防御**"——两者互不替代
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
    - Companion Bench human track（[`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md) §6.6）当前只是 RFC 中的 placeholder，没有 human annotator 招募协议 / 人评 UI / 人评数据 schema / 数据合规审查
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
  4. **Companion Bench human track 起步**（v0.3 时机）：
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
  7. **文档**：在 [`docs/external/eqbench3-submission-protocol.md`](external/eqbench3-submission-protocol.md) 加一段说明 streaming 启用条件；在 [`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md) Appendix A 同步注明 Companion Bench v0.1 不要求 streaming（保持 Companion Bench scope 干净）
- **优先级**：**中**（#29 P10 actuation 不需要；#30 修法 (1)/(3) 启动前必须做；建议在 Chatbot Arena 提交 packet 启动前 4-6 周开始）

## 32. Companion Bench v1.0 工程已就绪但 launch 路径未完全启动（working group / reference 实跑 / DNS / submission queue）

- **路径**：
  - Companion Bench v1.0 reference impl 已 land：[`packages/companion-bench/`](../packages/companion-bench/) Apache 2.0 wheel + 24 public + 96 held-out scenarios + 10 reference systems orchestrator + 9-page eqbench-parity 静态站 + 4 个 CI workflow + 145/145 测试全绿
  - **2026-05-11 update（rebrand + site landed packet）**：sub-track 4 / 5 部分闭合 — `site/CNAME` → `companion-bench.org` 已就位（DNS + Pages 设置仍需手动完成）；[`.github/ISSUE_TEMPLATE/{submission-request,bug-report,config}.yml`](../.github/ISSUE_TEMPLATE/) 已就位（独立 repo 拆分 + bot triage 仍未做）；详见 [#38](#38)
  - **缺位面**（v1.0 launch 必须项，目前仍是 placeholder）：
    - **Working group 未形成**：[`docs/external/companion-bench-governance-charter-draft.md`](external/companion-bench-governance-charter-draft.md) 章程模板就位但「Chair-elect candidates」段是空表；RFC §11 要求 ≥ 3 个组织 / 不超过 1 票 / rotating chair / 每季度 review
    - **`companionbench/heldout` 私有 repo 未创建**：[`.gitmodules`](../.gitmodules) 引用了它（路径仍是 `external/companionbench-heldout/` 保留 git-history 连续性；`heldout_loader` 自动接受 `external/companion-bench-heldout/` alias）但 repo 本身不存在；CI release-tier deploy key 未 provisioned；[`docs/external/companion-bench-heldout-bootstrap.md`](external/companion-bench-heldout-bootstrap.md) 一次性 organiser 步骤未执行
    - **真 10 reference systems 跑分未发生**：[`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) + [`scripts/companion_bench/run_companion_bench_paper_suite_full.sh`](../scripts/companion_bench/run_companion_bench_paper_suite_full.sh) 已就位但需要 ~$500-3500（single-seed）/ ~$5-15k（triple-seed）API 预算 + 6 个 vendor API key 才能产生真数据；当前 [`site/data/aggregate_results.json`](../site/data/aggregate_results.json) 由本 packet 新增的 [`scripts/companion_bench/populate_demo_site.py`](../scripts/companion_bench/populate_demo_site.py) 用 deterministic-fake 跑 8 系统 × 24 scenario × 1 seed = 192 arc 真实形状的 placeholder 填满（标 `demo: true`，site banner 显示）
    - **`companion-bench.org` 域名 / GitHub Pages CNAME 未真正生效**：`site/CNAME` 文件已写 `companion-bench.org` + workflow 已 deploy 整站；但**域名注册 + DNS A 记录 + Pages 自定义域名设置 + HTTPS 自动签发**这四步 IRL 未做（registry + DNS provider 操作，~30 分钟）
    - **Submission queue triage 未对外开放**：本 packet 已落 issue templates；RFC §11 + [`docs/external/companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md) §10 现在以「open issue with submission-request template」为入口；但**该 repo 尚未从 monorepo 独立**（仍在 VolvenceZero/VolvenceZero monorepo 内），没有 GitHub Actions bot 自动 triage（manifest 校验 / attestation 校验 / 自动跑 [`run_real_submission.py`](../scripts/companion_bench/run_real_submission.py)）
- **问题**：v1.0 reference impl 在 git 层完成了「方法学 + 代码 + 文档 + scenarios + scripts + CI」全套——但**「Companion Bench 是一个 live community benchmark」这件事**还没在外部世界发生。只要这些 placeholder 还在，对外说"Companion Bench v1.0 已发布"就是技术上正确、运营上空洞——任何记者 / 投资人 / 竞品研究员一搜到自定义域名 404、leaderboard 满是 demo 数据、charter draft 空候选人段，会立刻把 Companion Bench 归类为 vaporware vs 真 RFC convener。
- **违反**：不违反 R 铁律。这是**运营 / 治理 / 预算 / 域名层**债，不是代码层债。
- **风险**：
  - **短期低**：v1.0 reference impl 自身可被任何外部复用；任何团队都可以 clone repo 跑自己的 submission 拿自己的分数（RFC §3 P3 self-serve reproducibility 已成立）
  - **中期高**：[#29](#29) P10 actuation（EQ-Bench 公开提交）一旦完成、给我们留下「客观分数」之后，下一步 marketing / 战略合作 narrative 就会强 push「Companion Bench convener 身份」——本债不解决，convener 叙事就空了
  - **长期决定品类话语权**：陪伴 AI 这个品类如果在 2026-2027 谁能定义 evaluation 范式很可能就是谁的话语权；working group 不形成 → Companion Bench 永远停留在 v0.x 学术草案；working group 形成 → 是 Companion Bench 还是某个竞品定义出的 benchmark 成为产业 reference 的关键
- **触发条件**：
  - (a) 任何一次正式融资尽调材料准备（投资人会问「你们提的 RFC 有谁在用」）
  - (b) [#29](#29) P10 公开 EQ-Bench 提交后的市场推广窗口（48 小时内必须有 Companion Bench convener 故事跟上，否则只剩"我们在 EQ-Bench 第 N 名"这一条）
  - (c) 任何竞品（OpenAI / Anthropic / Meta / character.ai 类）发布自己的 long-session benchmark RFC（一旦发生，convener 窗口立即关闭）
  - (d) [#30](#30) Chatbot Arena 提交流程启动后需要一个对外 leaderboard URL 给 PR / 媒体引用
  - (e) 媒体 / 行业分析师专题（"How is companion AI evaluated?"）需要一个 live URL
- **推荐修法**（5 个独立可推进 sub-track）：
  1. **真 reference run**（~1 周 + 预算 approval）：BFC（Budget-First-Credible）—— 先批 ~$200-400 budget 跑 [`scripts/companion_bench/run_companion_bench_paper_suite_small.sh`](../scripts/companion_bench/run_companion_bench_paper_suite_small.sh)（公开 set + 1 paraphrase seed × ~5 reference systems），用 cost telemetry 校准 RFC §6.7 价格表后再决定是否批 release tier ~$5-15k；产物自动 push 到 [`site/leaderboard/data/aggregate_results.json`](../site/leaderboard/data/aggregate_results.json) 替换 demo
  2. **Held-out repo 创建 + deploy key**（~半天）：跑 [`docs/external/companion-bench-heldout-bootstrap.md`](external/companion-bench-heldout-bootstrap.md) 步骤；在 GitHub 创建 `companionbench/heldout` private repo + push 96 seed scenarios + 注册 deploy key；配 `COMPANION_BENCH_HELDOUT_DEPLOY_KEY` repo secret
  3. **Working group 形成**（~3-6 个月，组织层 follow-up）：识别 ≥ 2 个外部 align 的组织（建议候选：EQ-Bench 维护者 / RP-Bench 维护者 / 一家学术 AI 安全 lab / 一家 companion-class 产品公司）；走 [`docs/external/companion-bench-governance-charter-draft.md`](external/companion-bench-governance-charter-draft.md) 的 chair-elect 流程；填空候选人段
  4. **域名 + GitHub Pages CNAME**（~半天 IRL）：~~`site/CNAME` 已写~~（已就位）；剩下：注册 `companion-bench.org` 域名；DNS 配 `@` → `185.199.108.153` 等 GitHub Pages IP（4 条 A 记录）；GitHub Pages 设置自定义域名 + HTTPS 自动 issue；workflow [`.github/workflows/companion-bench-publish.yml`](../.github/workflows/companion-bench-publish.yml) 已 deploy 整站
  5. **Submission queue infra**（~1 周）：~~Issue template 三件套（submission-request / bug-report / config）已落~~（已就位 [`.github/ISSUE_TEMPLATE/`](../.github/ISSUE_TEMPLATE/)）；剩下：把 companion-bench 拆成独立 public repo（保留 monorepo 内的 dev 路径作为 internal mirror 或 archive）；加 GitHub Actions bot 自动 triage（manifest 校验 / attestation 校验）；submission 被 bot 接受后自动跑 [`scripts/companion_bench/run_real_submission.py`](../scripts/companion_bench/run_real_submission.py)（需要 self-hosted runner with API keys 或限速 cloud runner）
- **优先级**：**高**（不阻塞代码 land，但**直接阻塞** v1.0 → 真 launch 的转化；建议在 [#29](#29) P10 actuation 之后立即启动 sub-track 1+2+4，sub-track 3+5 与 [#30](#30) 推进节奏同步）

## 33. Companion Bench human-eval 轨道（RFC §6.6 / §6.6 + §11 核心承诺）实现 0%

- **路径**：
  - RFC [`companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md) §6.6 承诺一条 parallel human-eval track；RP-Bench 现状已经实证人评 Elo 与 LLM-judge Elo 系统性分歧 → 这条轨道是 Companion Bench 区分于 EQ-Bench 的最强承诺
  - 当前实现层完全空白：没有 annotator 招募协议 / 没有人评 UI / 没有 IRB-equivalent review / 没有 pairwise 投票 schema / 没有 anonymisation 规范
  - 与 [#30](#30) 修法 (4)（Companion Bench human track v0.3 起步）实质同一件事但视角不同：[#30](#30) 是**对外人评 footprint**，本债是**作为 Companion Bench benchmark 一部分的人评协议**
- **问题**：Companion Bench v1.0 reference impl 全部 LLM-judged。RFC §8.1 列出 LLM-judge 的 family bias / verbosity bias / formatting bias 三类已知威胁；§6.6 明确说"following RP-Bench's finding that human and LLM-judge rankings can diverge meaningfully, we treat human Elo as an independent measurement, not as ground truth"——这意味着**没有 human Elo 列的 Companion Bench leaderboard 就缺了 RFC 承诺的一半**。当前 [`site/leaderboard/index.html`](../site/leaderboard/index.html) 表格已为 human Elo 留了 placeholder column 但内容永远空。
- **违反**：不违反 R 铁律。但**违反 RFC §6.6 自身的承诺**——v1.0 不实现等于 RFC 文本与 reference impl 不一致。
- **风险**：
  - **短期低**：[#32](#32) sub-track 1（真 reference 跑分）跑完后 leaderboard 已经有可看的数；human Elo 缺失短期内不显眼
  - **中期高**：[#30](#30) 推进 + [#32](#32) sub-track 5（submission queue）启动后，「Companion Bench 自己也没有 human eval 你凭什么说我们超越了 LLM-judge bias」这条质问立刻浮现
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
  4. **TrueSkill / BT 求解器复用**（~1 天）：[`packages/companion-bench/src/companion_bench/elo.py`](../packages/companion-bench/src/companion_bench/elo.py) 已经有 `compute_trueskill` / `compute_bradley_terry`；用于 human Elo 列计算（参数与 LLM-judge 列保持一致以便对照）
  5. **leaderboard human Elo 列填充**（~半天）：[`site/leaderboard/leaderboard.js`](../site/leaderboard/leaderboard.js) 已 query `row.human_elo`；只需要 [`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 输出包含该字段；从人评数据计算
  6. **首次 study**（~1-2 月含 annotator 招募 + 标注期）：跑 ~10 个 reference systems × 全 24 public scenarios × 5 annotators / scenario，产出 Companion Bench v1.0 同步 first human-eval study 报告
  7. **守红线**：人评数据严禁回流 substrate / kernel；annotator 个人信息单独存与评分数据物理隔离；Companion Bench 永不发布个体 annotator id（聚合统计 only）；annotator agreement 明文禁止 LLM 代标
- **优先级**：**中-高**（不阻塞 v1.0 reference impl 的工程交付；但**阻塞** Companion Bench 长期作为「区分 LLM-judge / human-judge 的中立 convener」的核心叙事；建议与 [#30](#30) 修法 (4) 合并为一个 Companion Bench human-eval 工作组 sprint）

## 34. Companion Bench harness 性能 / 可扩展性 gap（顺序 SUT 调用 / 无 streaming / 无 retry / 无 staged executor）

- **路径**：
  - 当前 [`packages/companion-bench/src/companion_bench/sut_client.py`](../packages/companion-bench/src/companion_bench/sut_client.py) `OpenAIChatClient` 用 `urllib.request.urlopen` 同步调用，单次 timeout 默认 120s，无 retry / 无 backoff / 无 connection pool
  - [`packages/companion-bench/src/companion_bench/arc_runner.py`](../packages/companion-bench/src/companion_bench/arc_runner.py) `run_arc` 完全顺序——session × turn 都串行，单 arc 跑完需要 N×M 次往返
  - [`packages/companion-bench/src/companion_bench/submission.py`](../packages/companion-bench/src/companion_bench/submission.py) `run_submission` 顺序遍历 (specs × paraphrase_seeds)；24 × 3 = 72 arcs 串行
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
  - (b) 任何 release-tier 跑（[`run_companion_bench_paper_suite_full.sh`](../scripts/companion_bench/run_companion_bench_paper_suite_full.sh)）启动
  - (c) GitHub Actions 24h timeout 第一次撞墙
  - (d) Submission queue（[#32](#32) sub-track 5）启动 → 大量并发 submission 涌入
- **推荐修法**（按 ROI 排序）：
  1. **Async + connection pool**（~1 周）：把 [`sut_client.py`](../packages/companion-bench/src/companion_bench/sut_client.py) 从 urllib 改 aiohttp（已是 wheel 依赖）；arc_runner 改 async；同 session 内 turn 仍串行（保 cross-session-session_id 语义）但 multi-arc 并发；retry + exponential backoff on 429 / 5xx；token-bucket 速率限制 per-judge-model
  2. **Staged executor**（~3 天）：参 [`docs/implementation/10_pe_eta_dialogue_benchmark_harness.md`](implementation/10_pe_eta_dialogue_benchmark_harness.md) `run_real_dialogue_pe_eta_comprehensive_benchmark_staged` 模式；phase = (arc_run / callback_extract / perturn_judge / arc_judge / aggregate)；每 phase 落 manifest 到 `artifact_dir/manifest.json`；resume 时从已完成 phase 恢复
  3. **跨 system 并行**（~3 天）：[`score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 用 `concurrent.futures.ProcessPoolExecutor` 并发跑多 system（非 thread，因为 each system 子进程已是 IO-bound async）；上限按 judge 模型 quota 推算
  4. **SUT streaming**（与 [#31](#31) 同源）：companion-bench 当前 sync `chat()` 接收完整 response；如果 SUT 只支持 streaming（一些 vendor 推自己 SDK 时强制 streaming），companion-bench 直接断在 client 层。Adapter 改 streaming-aware 后，pre-emptively 解决这条路径
  5. **Cost-aware scheduler**（~1 周，optional）：[`cost.py`](../packages/companion-bench/src/companion_bench/cost.py) 已统计每 system / 每 judge / 每 axis 成本；加 budget cap（"一旦累计超 $X 暂停并 manual approve"）；CI release-tier workflow 接 budget guard
  6. **Checkpoint restart 测试**：mock 中途 SIGKILL 测 staged executor 能从断点续跑；用 in-memory mock SUT 不烧 token
- **优先级**：**中-高**（[#32](#32) sub-track 1 之后立刻启动；release tier 启动前必须做完 1+2+3）

## 35. Companion Bench 季度治理自动化（held-out paraphrase rotation / lexicon rotation / judge family rotation log）

- **路径**：
  - RFC §8.2 要求 held-out paraphrase seeds 季度 rotation；当前 [`scripts/companion_bench/generate_heldout_seeds.py`](../scripts/companion_bench/generate_heldout_seeds.py) 是 one-shot generator，没有 rotation salt 参数；同样的 input 永远产同样的 96 scenarios
  - RFC §6.3 + §8.1 要求 judge model 季度 rotation；[`docs/external/companion-bench-eqbench-crosswalk.md`](external/companion-bench-eqbench-crosswalk.md) 引用了 `companion-bench-judge-rotation-log.md` 但**该文件未创建**；charter draft 引用了 `companion-bench-heldout-rotation-log.md` 也**未创建**；[`docs/external/companion-bench-governance-charter-draft.md`](external/companion-bench-governance-charter-draft.md) §6 同样要求 hash-only 公开 diff 但格式未定
  - [`packages/companion-bench/src/companion_bench/lexicon.py`](../packages/companion-bench/src/companion_bench/lexicon.py) `LEXICON_VERSION = "1.0.0"` 是 string 常量；版本 bump 没有自动化路径，bump 之后 scenario_hash 是否要重生成也未定（lexicon 仅 runtime 消费，不进 hash，所以理论上 lexicon bump 不变 hash——但需要明文说清楚）
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
  - **长期高**：如果某次 rotation 被竞争团队 audit 出来从未发生，Companion Bench 作为 RFC convener 的可信度直接归零
- **触发条件**：
  - (a) Working group 形成（[#32](#32) sub-track 3）后第一个季度 review 会议
  - (b) 任何外部研究者或竞品质疑 "你们 RFC 说 rotation，repo 里看不到 rotation log"
  - (c) v1.1 release 节奏（quarterly cadence per RFC §9 implicit cadence）
- **推荐修法**：
  1. **`generate_heldout_seeds.py` 加 `--variant-salt` / `--rotation-quarter` 参数**（~1 天）：当 salt 变化时变体的 surface form 改变（persona tone / payload prefix / FSM seed offset 都依赖 salt）；保证同一 quarter 内所有人产出 byte-identical
  2. **rotation 自动化 GitHub Action**（~2 天）：`.github/workflows/companion-bench-quarterly-rotation.yml`，每季度 1 号 cron 触发；自动 (a) bump quarter salt (b) 重新生成 96 held-out scenarios 到 private submodule (c) push hash-only diff 到 `docs/external/companion-bench-heldout-rotation-log.md`（公仓）+ `external/companion-bench-heldout/HASHES.txt`（私仓）(d) open PR for chair sign-off (e) 不直接 merge——chair manual approve
  3. **judge rotation log 文件 + 自动 entry**（~半天）：创建 [`docs/external/companion-bench-judge-rotation-log.md`](external/companion-bench-judge-rotation-log.md)；每季度 working group 决议后由 chair 加一行 entry；CI workflow 在 [`scripts/companion_bench/run_companion_bench_paper_suite_small.sh`](../scripts/companion_bench/run_companion_bench_paper_suite_small.sh) / `_full.sh` 启动时校验「当前 judge model 与 log 最新一行一致」否则拒绝跑（防止"workflow 用 GPT-4 但 log 还写 GPT-5"漂移）
  4. **lexicon 版本协议**（~半天文档）：在 [`docs/specs/companion-bench.md`](specs/companion-bench.md) 加一节明确：lexicon bump 不影响 scenario_hash（已经如此）；lexicon bump 影响 identity slot 选择，但 [`draw_identity`](../packages/companion-bench/src/companion_bench/lexicon.py) 已经把 `LEXICON_VERSION` 包进 seed string 所以 paraphrase 仍 deterministic；新加 entries 是 backward-compatible，移除 entries 是 breaking
  5. **rotation log 模板**（~1 小时）：定义 `companion-bench-heldout-rotation-log.md` 与 `companion-bench-judge-rotation-log.md` 格式（quarter / chair-sig / hash diff / 96-count assertion）
- **优先级**：**中**（v1.0 launch 时不需要；但**必须在 working group 形成的同一季度内** land sub-track 1+2+3，否则第一次 rotation due 时会非常尴尬）

## 36. Companion Bench v2.x 长尾路径（multi-modal / EQ-Bench prompt 1:1 reuse / 加密 attestation / transcript 隐私）

- **路径**：4 个低优先但实质性的工程债，都与 Companion Bench v1.0 reference impl 已交付的代码相关，但都是 v2.x roadmap：
  - **(a) Multi-modal extension** — RFC §10 OQ4 明确"v0.1 is text-only. v2.x roadmap?"；当前 [`spec.py`](../packages/companion-bench/src/companion_bench/spec.py) `ScenarioSpec.user_simulator.fsm` 只发 `text` payload；voice / 表情 / 多模态 SUT 完全不支持
  - **(b) EQ-Bench rubric prompt 1:1 reuse** — [`packages/companion-bench/src/companion_bench/judge_perturn.py`](../packages/companion-bench/src/companion_bench/judge_perturn.py) `_PROMPT_HEADER` 是我们改写的版本而非 EQ-Bench 3 上游 prompt 原文；RFC Appendix A 承诺"Criteria 1–7 are aligned with EQ-Bench 3 to enable cross-benchmark comparison"——技术上 criteria 名 + 含义对齐了，但 prompt 文本不是 byte-for-byte 一致，跨 benchmark 信号转移精度比理论值低
  - **(c) Submission attestation 加密签名** — [`packages/companion-bench/src/companion_bench/submission.py`](../packages/companion-bench/src/companion_bench/submission.py) `SubmissionAttestation` 仅是 4 个 bool；submitter 把它们设 `true` 没有任何加密保证或时间戳；任何争议（"你们去年 12 月真的没用 Companion Bench derivative 训练吗"）无法事后审计
  - **(d) Transcript 隐私 / leaderboard 发布脱敏** — `arc_runner.py` 写出的 [`{arc_id}.bundle.json`](../packages/companion-bench/src/companion_bench/arc_runner.py) 包含 `identity_slot.name / occupation / contextual_detail`（来自 lexicon）+ 完整 user/assistant transcript；release-tier workflow 把整个 `artifacts/companion-bench/reference/` 上传为 GitHub artifact；这些**虽然不是真人 PII**（lexicon 是合成的），但 SUT 的 response 可能 inadvertently 泄露 vendor system prompt / 内部 vocabulary——leaderboard publication 应该有审查环节
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
  2. **(b) EQ-Bench prompt 1:1 reuse**（~1 天）：找 EQ-Bench 3 上游 rubric prompt 原文（Apache 2.0 / MIT 之类的兼容 license 下），把 [`judge_perturn.py`](../packages/companion-bench/src/companion_bench/judge_perturn.py) `_PROMPT_HEADER` 中前 7 个 criterion 的描述改为 EQ-Bench 3 verbatim；保留 Companion Bench 8th criterion（boundary_appropriateness）作为 Companion Bench 独有部分；在 [`docs/external/companion-bench-eqbench-crosswalk.md`](external/companion-bench-eqbench-crosswalk.md) 注明"prompt verbatim reuse since v1.x"
  3. **(c) Attestation signing**（~3 天）：`SubmissionManifest.attestation` 加 `signed_at: datetime` + `signature: str`（HMAC-SHA256 over manifest body 用 submitter 提供的 ed25519 key）；leaderboard 公开 submitter 的 public key 给 audit；争议时可加密验证 attestation 时间戳
  4. **(d) Transcript 隐私 review pipeline**（~1 周）：release-tier 把 `{arc_id}.bundle.json` 上传前先跑一个 anonymisation pass（去除任何疑似 system prompt 漏出 / 任何 PII-shaped 字符串）；leaderboard 公开版本只含 axis scores + scenario_hash，**不含** transcript body；transcript 留在 organiser 私有 storage（同 held-out 一样的治理）
- **优先级**：**低-中**（4 条都是 v2.x；(c) (d) 在 [#32](#32) sub-track 5 启动同时建议 land；(b) 任何时候都可以；(a) 是真 v2.0 工作量，等 multi-modal 商业化稳定后启动）

## 37. EQ-Bench 3 P10 actuation 未执行（真 Qwen substrate + 真 judge API + 三轨 ablation → 公开提交 verdict）

- **路径**：
  - [#29](#29) P1-P9 + wiring dry-run 已 land（2026-05-11，见 [`docs/external/eqbench3-wiring-evidence.md`](external/eqbench3-wiring-evidence.md)）：上游 `eqbench3` harness clone + `lifeform-openai-compat` wheel + 三轨 runner [`scripts/external_bench/run_eqbench3.py`](../scripts/external_bench/run_eqbench3.py) + verdict gate [`compare_ablation.py`](../scripts/external_bench/compare_ablation.py) + cross-walk 文档全套就位；synthetic vertical 上 45/45 scenarios + 26 debriefs 端到端跑通；adapter-side 隐藏 bug 清零（修复 `TEST_API_URL` 必须含 `/chat/completions` + PowerShell BOM 两条 wiring-only 问题）
  - **缺位面**（一次性 actuation，跑完即闭合本债）：
    - 没有真 Qwen 1.5B/7B 跑分：当前本地 torch 是 CPU-only（`2.11.0+cpu`），没下载过 `Qwen/Qwen2.5-1.5B-Instruct` 权重；synthetic vertical 的 EQ 分数不能代表真实 substrate 表现
    - 没有真 Anthropic `JUDGE_API_KEY`：`.env` 当前是 dry-run 占位 (`sk-ant-DRY-RUN-NO-REAL-CALL`)
    - 三轨 ablation（`companion / companion-cold / raw`）一次都没有真跑
    - `compare_ablation.py` 没有真 `.summary.json` 输入 → 没有任何 GO / HOLD verdict
    - 公开提交（[`docs/external/eqbench3-public-submission-checklist.md`](external/eqbench3-public-submission-checklist.md)）gate 在 verdict ≥ GO 上，本债不闭合就永远卡在闭门
  - 与 [#34](#34) Companion Bench harness 性能 gap **不同**：本债是 EQ-Bench（外部 benchmark）的实跑分，#34 是 Companion Bench（自定义 benchmark）的 harness 工程债；两者复用同一个 `lifeform-openai-compat` adapter wheel
- **问题**：[#29](#29) 的核心诉求是「投资人尽调时第一个 google 到的客观分数」。wiring 跑通 ≠ 拿到分数。当前对外仍无任何公开可引用的 EQ-Bench 3 数字；竞品（哪怕是套壳 Claude）只要交一次分就反超我们的 fundraising narrative。`compare_ablation.py` 守门的四条红线（frozen substrate / no kernel mod / no benchmark text in system prompt / no internal vocab in model card）已 statically 校验，但**实测前不知道**真分数是否过 `_DEFAULT_PUBLISH_THRESHOLD = 65.0`，因此不知道走 GO（公开提交 + 路径 (3)）还是 HOLD（保留内部 baseline + 转更大 substrate）
- **违反**：不违反 R 铁律。本债是预算 / GPU / 运营层 follow-up，与 vz-* / lifeform-* 内核架构正交。
- **风险**：
  - **短期低**：[#29](#29) wiring 已通过，知道实测时不会再撞 adapter-side bug
  - **中期高**：[#29](#29) 触发条件全部仍在生效——任何融资 / 战略合作尽调 / 媒体被问「your EQ-Bench score?」时本债是硬阻塞；wiring 已就位让「拿不出分」从「不知道能不能跑」升级为「明知能跑但没花钱跑」，叙事压力更大
  - **长期高**：[#30](#30) Chatbot Arena 提交 / [#32](#32) Companion Bench launch 都假设我们已经有 EQ-Bench 数字垫底——本债不闭合，整个对外 benchmark 故事的 narrative spine 缺一节
- **触发条件**：
  - (a) 任何一轮融资（A 轮及以后）尽调材料准备启动
  - (b) 战略合作方技术 due diligence
  - (c) 媒体 / 行业分析师专题被 cited 时被问 EQ-Bench 分数
  - (d) [#30](#30) Chatbot Arena 提交流程启动前
  - (e) [#32](#32) Companion Bench launch 推进到 sub-track 1（真 reference 跑分）前——逻辑上 EQ-Bench 应在 Companion Bench public leaderboard 之前出分，否则自定义 benchmark 看上去像规避客观评估
  - (f) 任何竞品交分进 EQ-Bench 3 leaderboard top 20（每周可能发生）
- **推荐修法**（一次性 actuation，按 ROI 排序）：
  1. **单轨真跑 calibration**（~半天 GPU + ~$2-5 judge）：先只跑 `companion` 一轨拿 wallclock + cost + 分数。命令：
     ```bash
     # shell 1: 真 Qwen substrate
     python -m lifeform_service.cli --host 127.0.0.1 --port 8770 \
       --vertical companion \
       --substrate-mode hf-shared --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \
       --substrate-device cuda \
       --enable-openai-compat
     # shell 2: 真 judge
     cd external/eqbench3
     export JUDGE_API_KEY=...  # 真 Anthropic key
     python eqbench3.py \
       --test-model lifeform-companion-qwen2.5-1.5B \
       --model-name vz-companion-calibration \
       --judge-model anthropic/claude-3.7-sonnet \
       --runs-file ../../artifacts/external_bench/calibration.runs.json \
       --threads 1 --iterations 1 --no-elo --ignore-canonical
     ```
     - 拿到分数后判断：(i) wallclock 是否符合预期（CPU baseline 14:42 + 真 Qwen 推理 ~5-10x 慢 → 估 1.5-2.5 h on GPU；CPU 跑会是 8-16 h）；(ii) judge cost 是否在 ~$1.50 / track 范围内；(iii) 分数大致档位（< 50 / 50-65 / > 65）
  2. **三轨完整 ablation**（~3-9 h GPU + ~$5 judge，单卡 24GB 串行；多卡 1.5-3 h 并行）：跑通后用 [`scripts/external_bench/run_eqbench3.py`](../scripts/external_bench/run_eqbench3.py) 三轨一把（`companion,companion-cold,raw`），产 3 个 `.summary.json` + attestation block
  3. **verdict 触发**（~5 min）：跑 `python scripts/external_bench/compare_ablation.py --summaries artifacts/external_bench/eqbench3_*.summary.json --output artifacts/external_bench/verdict.json` → 拿 GO / HOLD / insufficient_data
  4. **GO 路径**（验证 + 公开提交，~1 周）：(a) 走 [`docs/external/eqbench3-public-submission-checklist.md`](external/eqbench3-public-submission-checklist.md) 全 8 项；(b) 提交到 [eqbench.com](https://eqbench.com/) leaderboard；(c) `--with-elo` 跑 ELO pass（增量 ~$30-60/track × 3 = ~$90-180 总）；(d) 用 verdict.json 做 [`docs/external/eqbench3-results-internal.md`](external/eqbench3-results-internal.md) 内部 evidence；(e) 推 [#30](#30) Chatbot Arena 提交准备
  5. **HOLD 路径**（分数 < 65，~2 周）：(a) 不公开提交；(b) 留作 internal baseline；(c) 尝试更大 substrate（Qwen2.5-7B-Instruct 或 14B）重跑路径 1+2+3；(d) 同步在 [`docs/external/eqbench3-eqbench-crosswalk.md`](external/eqbench3-eqbench-crosswalk.md) 注明 substrate 切换原因；(e) 评估是否值得提升 substrate fingerprint 兼容性（vs 继续走 small-companion-relational 叙事）
  6. **insufficient_data 路径**（任何 track 拿不到 rubric_average）：检查 `compare_ablation.py` 报错 → 多半是 [#31](#31) streaming SSE 或 lifeform-companion 真 Qwen 路径上的某个回调没 wire 进 OpenAI-compat 响应；本债转为 [#29](#29) follow-up + 修复后重跑
  7. **守红线**：所有 4 项 attestation (`frozen_substrate / no_kernel_modification / no_benchmark_text_in_system_prompt / no_internal_architecture_terms_in_model_card`) 已 statically 校验；提交前 review 一次 model card 文字确保对外抽象（"long-context companion system with adaptive memory"）正确，不出现 NL/ETA/R-PE/regime/owner SSOT/F1-F6 任何内部术语
- **优先级**：**中-高**（不阻塞代码运行，但**直接阻塞** [#29](#29) / [#30](#30) / [#32](#32) 的对外叙事链路；建议在下一轮融资 kickoff 前 2-3 周启动单轨 calibration，再 1-2 周完成三轨 + verdict + 提交）

## 38. Companion Bench 公开站点 v1.0 上线后的小尾巴（rebrand cleanup / verifier / demo realism / build_site 增量化）

- **路径**：2026-05-11 packet 把 Companion Bench → Companion Bench rename + 9-page eqbench-parity site + `build_site.py` + `populate_demo_site.py` + `compare.html` + `judges.html` + Issue templates 全套 land；145/145 测试全绿，site URL 22/22 200。下面 6 条是这个 packet 显式 deferred 的小尾巴，单独都不阻塞 launch，但堆积起来会让 v1.0 站点在第二个季度 review 时显得粗糙。
  - **(a) 7 个 Companion Bench redirect stub 未删**：`docs/external/companionbench-{rfc-v0,submission-protocol,governance-charter-draft,eqbench-crosswalk,heldout-bootstrap,public-scenario-hashes}.{md,txt}` + `docs/specs/companion-bench.md` 留作 5-line redirect 防 404。一发 release 后（外部最后一次抓取 ~ 2026-Q3）即应删除；删之前在 `docs/external/eqbench3-submission-protocol.md` / 任何 lifeform-* 文档里再 grep 一遍 `companionbench-` 字面引用。
  - **(b) `pyproject.toml` 残留 `companionbench` keyword + description footnote**：[`packages/companion-bench/pyproject.toml`](../packages/companion-bench/pyproject.toml) `keywords` 里同时保留 `"companion-bench"` 与 `"companionbench"`（PyPI 旧名搜索兼容）；下次 minor bump 时移除 `companionbench`。同样 `description` 里的 "Previously circulated as Companion Bench" footnote 也保留至少一年再清理（PyPI 描述长尾搜索）。
  - **(c) 每 submission detail page 的 `verifier.state` 永远是 `"pending"`**：[`scripts/companion_bench/build_site.py`](../scripts/companion_bench/build_site.py) `build_submission_detail` 把 `verifier` 字段硬写 `"pending"`，没有自动化 re-run hook。RFC §7.3 承诺「organisers re-run one random public-test arc per submission to verify reproducibility」+ [`packages/companion-bench/src/companion_bench/verifier.py`](../packages/companion-bench/src/companion_bench/verifier.py) `pick_verification_arc` 已实现 deterministic arc-picking 但**未接到 build_site / CI workflow**；所以每个 detail page hero 永远显示 "verified by re-running one random arc; report **pending**"。需要：(i) `score_reference_systems.py` / `run_real_submission.py` 跑完后调 `verifier.run_verification(...)` 跑一次 verifier-tier arc + diff axis 分；(ii) build_site 读 verifier output 改写 `verifier.state` 为 `"pass" / "flagged" / "missing"` + 写入 verified-arc-id + axis-diff%。
  - **(d) `populate_demo_site.py` 出来的 8 系统分数全在 50-52 区间**：因为 [`DeterministicFakePerTurnJudge`](../packages/companion-bench/src/companion_bench/judge_perturn.py) + [`DeterministicFakeArcJudge`](../packages/companion-bench/src/companion_bench/judge_arc.py) 的 deterministic seed 跟 SUT 输出 prefix 几乎无关——demo prefix 改 8 种但 fake judge 仍出近似分。结果 leaderboard 看起来"所有系统都在 50 附近排队"——非常不像真的 benchmark。**目前**靠 `demo: true` banner 自描述；**真 reference run** 落地后（[#32](#32) sub-track 1）会自然修复。如果在 sub-track 1 之前需要给 demo 再加一层 visual realism，可以在 fake judge 里加 per-system noise term（hash(system_id) → ±15 分），但**不要混进真 judge 路径**——只在 `populate_demo_site.py` 里 monkey-patch fake judge。
  - **(e) `site/data/judge_calibration.json` 是 illustrative 数据**：本 packet 写了 6 axis × Spearman 0.69-0.83 + 12 calibration scatter point，标 `demo: true`。真 judge calibration 数据要等 [#33](#33) sub-track 1+2（annotator 招募 + 人评 UI）跑完第一轮人评 study 才能产生。在那之前 `judges.html` 读到的就是这份 illustrative 数据；UI 层显示是 OK 的，但这个文件**不能** ship 到独立 PyPI / archive 因为它会被外部当做真 calibration evidence 引用。
  - **(f) `build_site.py` 是全量重建，无 incremental / no checkpoint**：单次 `python scripts/companion_bench/build_site.py --artifact-dir <X>` 无论改了几个 submission，都把 `site/data/aggregate_results.json` + 所有 `site/data/submissions/*.json` + `pairwise.json` 全部 re-emit。对 8-10 个 submission 的 demo 跑 ~10 秒 OK；对 [#32](#32) sub-track 1 之后的 ~50 个 submission（10 reference + 持续接收 community submission）会拖到 ~分钟级，且每次 commit `site/data/` 全 diff。建议：(i) 若 manifest 里 `submission_id` 已存在且 artifact 时间戳未变 → skip；(ii) 只重算受影响 submission 的 `submissions/<id>.json` + 全局 `aggregate_results.json` + `pairwise.json`（pairwise 必须全算因为 TrueSkill / BT 都是 global）。
- **问题**：单看每条都是小事，但**累加起来定义了"Companion Bench 公开站点的工程精度"**：detail page 永远 pending、demo 数据看起来像一团泥、calibration 数据是合成的、redirect stub 还在历史路径——任何审计员或竞品分析师 5 分钟扫一遍都会得出"这是个仓促 ship 的 alpha"印象，与 RFC 承诺的 v1.0 release 调性不符。
- **违反**：不违反 R 铁律。这是**公开 polish 债**。
- **风险**：
  - **短期低**：每条单独都不阻塞 [#32](#32) sub-track 1（真 reference run）落地
  - **中期中**：[#32](#32) sub-track 1 完成 + 媒体 / 学术引用 site 时被注意到（"为什么所有 demo 数据都聚在 50 分" + "为什么 verifier 永远 pending"）
  - **长期低-中**：(a) (b) 一年后没人还在引用旧 `companionbench-*.md` 路径；(c) (d) (e) 都跟 [#32](#32) / [#33](#33) 推进节奏一起自然消解；(f) 只有 submission queue 真热起来才会暴露
- **触发条件**：
  - (a) / (b)：[#32](#32) v1.0 → v1.1 release 节奏（~2026-Q3 自然窗口）
  - (c)：[#32](#32) sub-track 1（真 reference run）落地后立刻浮现（detail page 显示 "pending" 但其实已经跑过真 verifier 就尴尬）
  - (d)：[#32](#32) sub-track 1 推进前如果有 demo screenshot 出现在媒体 / blog post 里
  - (e)：[#33](#33) sub-track 1 之前如果有人引用 `judge_calibration.json` 当真 calibration evidence
  - (f)：[#32](#32) sub-track 5（submission queue）真启动后第一次大批 submission 涌入
- **推荐修法**（按 ROI 排序，每条 ≤ 1 天）：
  1. **(c) verifier 自动化 wire-up**（~1 天）：[`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 跑完每个 system 后调 [`verifier.run_verification(submission_id, artifact_dir, ...)`](../packages/companion-bench/src/companion_bench/verifier.py)；verifier 写 `<submission_id>/verifier_report.json`；[`build_site.py`](../scripts/companion_bench/build_site.py) `build_submission_detail` 读这个 report 填 `verifier` 字段。优先级最高因为 detail page 直接读这个字段
  2. **(f) build_site 增量化**（~半天）：用 artifact dir mtime 或 manifest hash 做 skip 判断；提供 `--full` 强制全量；CI workflow 默认增量
  3. **(d) demo 视觉真实化**（~半天）：`populate_demo_site.py` 加 per-system axis noise（基于 `hash(submission_id)` 派生 ±10-15 分 deterministic noise），让 demo 看起来像真 benchmark；不改 fake judge 本身
  4. **(a) redirect stub 清理 PR**（~1 小时）：~2026-Q3 一发 release 后单独 PR 删 7 个 stub；commit message 链接到 [`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md) footnote
  5. **(b) `companionbench` keyword + footnote 清理**（~10 分钟）：~2027-Q1 minor bump 时一并改
  6. **(e) judge_calibration.json 改为真数据**（依赖 [#33](#33) sub-track 1+2 完成）：[`scripts/companion_bench/`](../scripts/companion_bench/) 加 `score_judge_calibration.py` 把人评 study 输出转成 `judge_calibration.json` schema
- **优先级**：**中-低**（[#32](#32) sub-track 1 是入口；本债的 6 条按 ROI 排序，1-3 在 sub-track 1 同窗口完成；4-5 等 release 节奏；6 等 [#33](#33) 推进）

---

## 39. Wave K curated bundle 的 `coverage_map` 过严：in-corpus 题被 L4 ScopeRefuser 当 OOS 拒掉

- **路径**：Wave Q dry-run（`artifacts/figure_verify/einstein-tinygpt2-curated/transcript.md`）显示，针对 Wave K curated bundle 的真 corpus chunk 派生的 in-corpus 立场题 —— prompt 形如 *"Speaking from your own primary writings, what is your perspective on the relationship between principal, considerations, postulate, relativity?"* —— 在 `BUNDLE` / `BUNDLE_LORA` 条件下被 L4 短路到 `"I'm sorry — that topic falls outside what this figure documented in their primary sources."`，rationale tag `l4_scope_refusal`。可这些题的 ground-truth chunk 就是这个 bundle 自己的 paper（`paper:wikisource:en:the_foundation_of_the_generalised_theory_of_relativity:...`）。这意味着 [`packages/lifeform-domain-figure/src/lifeform_domain_figure/coverage_map.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/coverage_map.py) 在真 Wave K corpus（2 篇 substantive paper, 444 chunks）上的 coverage scoring 阈值偏紧 —— 大概率是 hashing-embedding cosine 在小 corpus 上离散度过低 + `STRICT_REFUSE` 拍门阈值过高，导致 retrieval index 自己的 chunk 也通不过 coverage 自检。
- **问题**：直接结果是 `gate_cognition_improves` 永远 fail（响应文本是固定 refusal 模板，跟 GT chunk cosine 当然 0.00）+ `gate_evidence_emerges` 永远 fail（refusal 不调 GroundedDecoder）。**这不是验证管线的 bug —— 是 figure-vertical L4 enforcement 在真 curated bundle 上的真 production bug，被 verification harness 第一次跑就抓到了。**
- **违反**：不直接违反 R 铁律。这是 L4 实现 quality 缺陷：`coverage_policy=STRICT_REFUSE` 应该只对**严格 OOS** 短路，不应对**自己 corpus 涵盖的 topic** 短路。当前对真 corpus 的 false-positive 率太高。
- **风险**：
  - **短期高**：阻塞 `gate_cognition_improves` + `gate_evidence_emerges` 在真 Wave K bundle 上 PASS。在 fix 这条之前 verification verdict 永远 4-fail，无法证明 cognition 链路通。
  - **中期高**：DLaaS 上线 Einstein lifeform 时用户问 "what is your view on relativity" → 收到 "that topic falls outside" 是 product-breaking 体验。
  - **长期中**：相同的 coverage_map 阈值 logic 一旦复用到第二个 figure（比如 lu_xun），同 bug 复发。
- **触发条件**：本 dry-run 已触发；任何对真 curated bundle 跑 `verification.persona.cli` 都会触发。
- **推荐修法**（两条任选其一，或并行）：
  1. **降低 `STRICT_REFUSE` 阈值 / 提供更友好的回退** —— 给 `coverage_map.evaluate(query)` 加一档 `SOFT_DISCLAIM` 或 `LOW_CONFIDENCE_PASS`：当 query top-k retrieval 命中至少 1 个 chunk 但 coverage_score 在 `[soft_min, strict_min)` 之间时不短路，让生成跑完后由 L3 GroundedDecoder 决定是否标 evidence；只有当 retrieval **零命中** 才 STRICT_REFUSE。这条最快，~半天。
  2. **重新 calibrate Wave K bundle 的 coverage_map 构造**：[`build_figure_coverage_map`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/coverage_map.py) 在 444-chunk 真 corpus 上 emit 的 topic vector 太窄 —— 检查 `top_topic_terms` 是否被 stop-word + 通用 physics 词污染；扩 `term_list` 加上 corpus 显式频次最高的 50 个 physics 名词（"relativity / postulate / aether / inertia / equivalence / ..."）。~1 天。
  3. **回归测试**：[`packages/lifeform-domain-figure/tests/test_coverage_map_self_consistency.py`](../packages/lifeform-domain-figure/tests/test_coverage_map_self_consistency.py)（新）—— 给一个真 curated bundle，断言 *bundle 自己 retrieval_index 里 ≥ 80% 的 chunk locator 经过 coverage_map.evaluate 不会 STRICT_REFUSE*。这个 contract 加上后，本 debt 就不会再无声复发。
- **优先级**：**中-高**（直接阻塞 #41 真 Qwen 跑分时 gate 通过；推荐先做 (1) 快速救急 + (3) contract test，再排 (2)）

---

## 40. Synthetic LoRA backend 的常数 delta 经过 LayerNorm 被吃掉：BUNDLE 与 BUNDLE_LORA forward 不可区分

- **路径**：Wave Q dry-run 显示 `bundle.voice_score == bundle_lora.voice_score`（0.272 vs 0.272，Δ=0.0），原因是 [`SyntheticLoRABakeBackend.bake`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora.py) 派生的 adapter delta 是从 `training_plan_hash` 派生的**常数 vector**（per-layer 全位置同值）；tiny-gpt2 的 `LayerNorm` 在 forward 时会把 mean 减掉 → 常数 delta 进 LayerNorm 出来变 0。Wave D 实测时已经知道这点（[`packages/vz-substrate/tests/test_lora_aware_runtime_smoke.py::_aggressive_persona_layers`](../packages/vz-substrate/tests/test_lora_aware_runtime_smoke.py) 故意用交替正负 delta 才能产生可观测 logit shift）。
- **问题**：`gate_voice_improves_with_lora` 是这套验证里**载荷性 gate** —— 它专门用来证明 "LoRA 真改了 forward" —— 但在 synthetic backend 上永远 Δ=0.00，gate 永远 fail。production 路径（PEFT-trained LoRA on Qwen）不会有这个问题，但 SHADOW 部署 + CI 默认全是 synthetic backend，所以这个 gate 在默认配置下基本是死的。
- **违反**：不违反 R 铁律。是 synthetic backend 的可观测性缺陷：synthetic 的本意是 "确定性、CPU-friendly、跨机器 byte-identical"，但**没有要求 forward 上必须可观测**。
- **风险**：
  - **短期低**：`gate_voice_improves_with_lora` 在 PEFT backend + 真 Qwen 跑分时（[#41](#41)）会自动恢复正常；synthetic 路径只是看不到 LoRA 真改。
  - **中期中**：reviewer 看 SHADOW verdict 容易误判 "LoRA 没起作用 / pool 没注册" —— 看 `verdict.json.notes` 区分 BUNDLE_LORA fall-through 还是 forward 没观测到。
  - **长期低**：真上线一定走 PEFT，本 debt 自然消解。
- **推荐修法**：
  1. **Synthetic backend 派生 alternating delta**：把 [`SyntheticLoRABakeBackend.bake`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/lora.py) 的 delta 从常数改成"按位置交替正负 + 散布"（参考 Wave D 的 `_aggressive_persona_layers` 模式）；保持 `training_plan_hash` 派生 byte-identical 不变（同 hash → 同 delta），但 LayerNorm 后剩余可观测 shift。~半天。
  2. **`gate_voice_improves_with_lora` 加 `min_substrate_observability` 门**：当 backend 是 synthetic 时把 voice gate 标 `skipped:synthetic_backend_layer_norm_eats_delta` 而不是 `failed`；只在 PEFT backend 上要求 Δ ≥ threshold。这样 SHADOW verdict 不会假阴。~半天。
  3. **回归测试**：[`packages/vz-substrate/tests/test_synthetic_lora_post_layernorm_observability.py`](../packages/vz-substrate/tests/test_synthetic_lora_post_layernorm_observability.py)（新）—— 给 synthetic LoRA + tiny-gpt2，跑 forward 一次断言 logit Δ > 1e-6。配合 (1) 同时 land。
- **优先级**：**中**（不阻塞 [#41](#41) 真 Qwen 跑通；阻塞 SHADOW 路径下的 verdict 信号质量）

---

## 41. 真 Qwen-1.5B PEFT bake + verification 跑分未执行（cognition / evidence gate 通过的硬前提）

- **路径**：Wave N-P 装好的管线**理论上**支持真 Qwen-1.5B PEFT bake（[`scripts/figure_bake_einstein_persona_lora.sh`](../scripts/figure_bake_einstein_persona_lora.sh) 默认 env 已留 `QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct` + `PEFT_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj` slot），但 Wave Q dry-run 用 tiny-gpt2 (~10M params) 跑 —— 这台开发机 Win/CPU 跑不动 1.5B model 的 PEFT 训练 + 推理（峰值 6-8 GB RAM + 数小时 CPU 时间）。结果：`raw / bundle / bundle_lora` 三条 condition 在 in-corpus 题上 cognition_score 全 0、L3 evidence_count 全 0；tiny-gpt2 emit 的是 boilerplate fallback "I can stay with the current context..." 而不是 Einstein 内容。
- **问题**：cognition gate + evidence gate 通过的**硬前提**是 substrate 真能模型 Einstein —— tiny-gpt2 太小做不到，Qwen-1.5B 是最低门槛。本 debt 跟踪 "把 Wave N-P 管线在真 GPU + 真 Qwen 上跑一遍"。
- **违反**：不违反 R 铁律。这是**资源 / 运维债**，不是代码债。代码层 Wave N-P 已就位。
- **风险**：
  - **短期中**：在真 Qwen 跑分前，无法对外宣称 "Einstein vertical 真有 Einstein 的口气和认知" —— 只能说 "管线就位，等 GPU"。这是融资 / 招募侧的 evidence gap。
  - **中期高**：跟 [#37 EQ-Bench 3 P10 actuation](#37) 同样的 GPU 资源约束 —— 本债跟 #37 共享一台 GPU 时间；优先做哪个取决于 evidence-gap 优先级。
  - **长期低**：一旦做了一次完整跑分（见下方 reproduction recipe），随后只在 Wave K corpus 重大变动时才需重跑。
- **触发条件**：任何 GPU 实例（A10 / L4 / RTX4090 任一）+ 1 小时空闲就能跑一遍。
- **推荐修法**（reproduction recipe）：
  ```bash
  # 1. 真 Qwen + 真 PEFT bake（~30 min on 1× A10 / L4）
  QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
  PEFT_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj \
  PEFT_RANK=8 PEFT_MAX_STEPS=200 PEFT_DEVICE=cuda \
      bash scripts/figure_bake_einstein_persona_lora.sh

  # 2. 真 verify（~20 min on same GPU）
  RUNTIME_BACKEND=transformers \
  QWEN_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
  SKIP_BAKE=1 \
      bash scripts/figure_verify_einstein_persona.sh

  # 3. 看 verdict.json.gates[*].passed —— 如果 #39 已修，cognition + evidence 应能 PASS
  ```
- **依赖**：先做 [#39](#39)（coverage_map 过严修复）+ [#40](#40)（synthetic delta 修；不强依赖，但同步做更干净），再做本债。
- **优先级**：**中-高**（融资 / 招募 evidence 视角；GPU 资源到位后 ~1 小时可完成第一次真跑）

---

## 42. Persona verification refusal-precision 阈值 + 探针集合在 5 道样本上离散度过粗

- **路径**：Wave Q dry-run 显示 OOS refusal_rate = 0.60（5 道里 3 道触发 L4），低于 `gate_refusal_works.threshold = 0.80`。这里有两个层叠债务：(a) 探针只 5 道，1 道 = 0.20 量化步长 → 阈值 0.80 在 5 道样本上意味着至少 4/5，差 1 道就 fail（一个不稳定的 LLM 输出就能翻盘）；(b) [`OUT_OF_SCOPE_REFUSAL_QUESTIONS`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/out_of_scope_set.py) 的 5 道（tiramisu / sourdough / Python tutorial / car maintenance / pop song）跟 Einstein corpus 在 hashing-embedding 空间里有意外重叠 —— 比如 "Python" 跟 paper 里的 "Lyapunov" 共享 prefix；coverage_map 不会刚性区分。
- **问题**：refusal gate 在小样本 + 探针集合不够 sharp 时**无法稳定触发**。Wave Q 是 4-fail verdict 的一部分，但即使 [#39](#39) 修了 coverage_map 阈值，本债仍在。
- **违反**：不违反 R 铁律。这是**测试集 quality / 阈值统计 power** 债。
- **风险**：
  - **短期低**：单看 Wave Q 一次 dry-run，本债排在 #39 / #41 之后；refusal gate 不是核心信号。
  - **中期中**：如果做完 #39 + #41 后 verdict 仍卡在 refusal gate，reviewer 会怀疑 ScopeRefuser 整体不工作 —— 但其实只是探针集合 + 阈值标定问题。
  - **长期中**：跟 Wave P 已记录的 follow-up debt（gate threshold ROC 校准）合并 —— 等积累 ≥ 30 个 verdict run 后做 ROC 选阈值。
- **推荐修法**（按 ROI 排序）：
  1. **扩 OOS 探针到 ≥ 20 道**（~半天）：reviewer 写 15-20 道明显 off-corpus 的题，覆盖 culinary / software / automotive / entertainment / modern-tech / sports / fashion 至少 7 个 domain；每个 domain 至少 2 道防止单 domain 全错。这样 quantum step 从 0.20 降到 ≤ 0.05。
  2. **拆 `gate_refusal_works` 成 per-domain pass-rate**（~半天）：而不是 single 0.80 threshold，要求 ≥ 5 个 domain 各自 refusal_rate ≥ 0.50。这样 sharper signal、不被单 domain 拉偏。
  3. **ROC 校准**（依赖 (1)）：等 (1) 后跑 ≥ 30 个 verdict run（不同 figure / 不同 LoRA bake step / 不同 substrate 模型），统计 (refusal_rate, hand-graded "actually refuses correctly?") pair；用 ROC 选 threshold 而不是 0.80 hand-tune。
  4. **sharper hashing embeddings**：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/coverage_map.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/coverage_map.py) 当前用 hashing trick 派生 embedding；考虑改用 sentence-transformer mini-encoder（offline embedding，pin via integrity hash），让 "Python" vs "Lyapunov" 真有 cosine 区分度。这条跟 [#28 残余 reviewer 工艺](#28) 同窗口做。
- **优先级**：**低-中**（不阻塞 #39 / #41；但做完 #39 + #41 后如果 verdict 还卡在 refusal，本债成主路径）

---

## 43. Arch Uplift Phase 2 follow-up（6 项 Phase 1 deferred 工作 + cleanup triggers）

- **路径**：
  - 主 plan：[`docs/moving forward/experiment-arch-uplift.md`](moving%20forward/experiment-arch-uplift.md)（A1-A5 + B1-B4 完整 spec）
  - Phase 1 exit evidence：[`docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md`](moving%20forward/experiment-arch-uplift-phase1-exit-evidence.md) §4 列出 6 项 deferred
  - sub-spec 出口：[`docs/specs/profile-registry.md`](specs/profile-registry.md) / [`docs/specs/evaluation-cascade.md`](specs/evaluation-cascade.md) / [`docs/specs/audit-owner.md`](specs/audit-owner.md)
- **问题**：2026-05-13 落地的 Arch Uplift Phase 1 是 **schema + 接口 + 骨架 + contract test** 层；6 项实质实施仍未做，每项都是独立 packet：
  1. **A1 阶段 2**：[`packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py`](../packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py) legacy 11 个 profile 仍用 `if profile_label == "X"` 分支；**但 Phase 2 四条候选 profile 已接入 registry + dispatch**（`cpd-beta-switch` / `counterfactual-credit` / `tom-owner` / `persona-geometry-readout`），作为最小 SHADOW evidence 入口。剩余 debt 是把 legacy 11 个 profile 也迁到 registry-first dispatch。
  2. **A2 backbone 迁移**：[`packages/vz-cognition/src/volvence_zero/evaluation/backbone.py`](../packages/vz-cognition/src/volvence_zero/evaluation/backbone.py) compute_* helpers 仍在原文件，未搬到 `cheap_layer.py`；当前 `EvaluationCheapLayer` 仅是 marker facade。
  3. **A2 cascade 实际计算**：[`evaluation/mid_layer.py`](../packages/vz-cognition/src/volvence_zero/evaluation/mid_layer.py) 已能抽 COG-1 readout，`expensive_layer.py` 已能从 metric means 生成 deterministic head-to-head，`cross_generation_aggregator.py` 已能聚合 gate evidence。剩余 debt 是把这些从脚本 utility 推进到正式 runtime/cascade process 与真实 multi-seed aggregate。
  4. **A5 audit-agent 内容**：[`packages/vz-cognition/src/volvence_zero/audit/module.py`](../packages/vz-cognition/src/volvence_zero/audit/module.py) 已从 empty snapshot 推进到 readout-only probes（evaluation alert / persona drift / least-control effort）并发布 risk score。剩余 debt 是 N8 8-attack elicited probe 工具链（dataset inspector / benchmark runner / persona drift probe / memprobe runner）与 rare-heavy `audit_required=True`。
  5. **B2 阶段 2**：[`packages/vz-substrate/src/volvence_zero/substrate/adapter.py`](../packages/vz-substrate/src/volvence_zero/substrate/adapter.py) `feature_surface` / `residual_activations` 仍是 recommended 而非 abstract；contract test [`test_substrate_feature_hook_completeness.py::test_production_adapter_hook_population_report`](../tests/contracts/test_substrate_feature_hook_completeness.py) 当前是 informational SKIP（pre-promotion）。
  6. **B3 benchmark union**：`RuntimeModule.declare_benchmark_metrics()` 接口已建；dialogue / paper-suite benchmark 的 `metric_means` 仍主要是硬编码 key 集。**Phase 2 候选 readout 已手动接入 `metric_means`**（CPD / least-control / ToM interlocutor count / persona geometry），剩余 debt 是改为 `union(hardcoded, declared)` 的通用声明式抽取。
- **违反**：不违反 R 铁律。Phase 1 已经把所有不变量（R2 frozen base / R4 token 空间禁忌 / R8 SSOT / R12 evaluation = readout / R15 可回滚）都纳入 contract test 守门。这条债是**"骨架已立但还没装内容"**的工程残余，不是设计缺陷。
- **风险**：
  - **短期低**：Phase 1 全部默认 DISABLED（mid / expensive / aggregator）或 SHADOW empty（audit），现有 11 profile + 22 个 `test_credit_gate.py` + 现有 6 cheap_layer 下游消费者全部 byte-equivalent，不影响任何现有 functionality。
  - **中期中**：[#44 阶段 C 实验](#44) 起跑前必须完成至少 (1) A1 阶段 2 dispatch（否则候选 capability 不进 runtime）+ (6) B3 benchmark union（否则候选 readout 不进 metric_means）。
  - **长期中**：(3) A2 cascade 实际计算 + (4) A5 audit-agent 是 #44 决策机制最终成型的硬前置；不做则 ModificationGate 仍只能消费现有 evaluation snapshot（双门），三类证据中只接通一类。
- **触发条件**：
  - **legacy 11 profile 迁移到 registry-first dispatch** → 继续推进 (1)
  - **新增第五个阶段 C 候选或想去掉手写 metric list** → 继续推进 (6)
  - **COG-3 起跑**（persona drift readout 需要真 feature_surface）→ 触发 (5)
  - **OA-4 业务 packet 启动**（audit-agent 工具集） → 触发 (4)
  - **阶段 C 多 profile 对照** 想用 mid_layer 的 ablation delta → 触发 (3) 的 mid_layer 部分
  - **阶段 D 决策**（profile → ACTIVE 切换）想从 synthetic/single-run 升级到真实 multi-seed verdict → 触发 (3) runtime aggregate + (4) full audit-agent
  - **Arch Uplift "完整完成"判定**（[`experiment-arch-uplift.md`](moving%20forward/experiment-arch-uplift.md) §10 5 条退出条件）→ 触发全部 6 项
- **推荐修法**（按依赖图，可并行 / 串行明确）：
  1. **(1) + (6) 已有 Phase 2 最小切片**：四条候选已能显式跑 SHADOW profile，且关键 readout 已进 `metric_means`。剩余工作是泛化 legacy dispatch 与声明式 metric union。
  2. **(5) B2 阶段 2** 与 (1)(6) 可并行；只在 COG-3 起跑前完成即可。
  3. **(3) A2 cascade 实际计算** 分 3 阶段做：先 mid_layer aggregation（支持 COG-1 ablation delta）→ 再 expensive_layer head-to-head（支持 DM-7 winrate）→ 再 cross_generation aggregator（接 ModificationGate evidence）。可与阶段 C 各候选业务 packet 并行推进。
  4. **(4) A5 audit-agent** 走 OA-4 业务 packet 单独节奏；rare-heavy artifact 路径首先消费，dialogue-online 路径保持 `audit_required=False`。
  5. **(2) backbone 迁移** 是 cleanup 级别工作，可以等 cascade 跑稳定后再做；不阻塞 #44。
  6. **Phase 1 cleanup triggers**（来自 exit evidence §5）：legacy if-elif removal 等 (1) 阶段 2 + ≥1 release cycle 稳定；旧 `EvaluationModule` 路径 removal 等 (2) + (3) 完成；pre-A5 `evaluate_gate_reasons` default removal 等 (4) OA-4 + rare-heavy 路径切 `audit_required=True`。
- **优先级**：**低**（用户 2026-05-13 明确指示）。Phase 1 已建立完整的契约面 + 96 个新 contract test 守门，不做 Phase 2 不影响系统功能；只有当真正开始跑阶段 C 实验时才需要按上述依赖图 unblock。

---

## 44. 阶段 C 4 候选 SHADOW 实验 + 阶段 D 决策机制（建立在 Phase 1 之上的实验设计）

- **路径**：
  - 主规划：[`docs/moving forward/experiment.md`](moving%20forward/experiment.md) §4 阶段 C + §4 阶段 D + §6 候选起跑顺序
  - 现状核查：[`docs/moving forward/experiment-phase-a-brief.md`](moving%20forward/experiment-phase-a-brief.md)（每个候选的 owner / slot / 耦合 / 起跑前置详表）
  - 候选来源：[`docs/moving forward/探索方向.md`](moving%20forward/探索方向.md) §SYS-1 / §COG-1 / §COG-2 / §COG-3
  - shadow evidence harness 模板：[`scripts/run_shadow_evidence_template.py`](../scripts/run_shadow_evidence_template.py)
- **问题**：阶段 A 现状核查完成 + Phase 1 架构地基已铺好；截至 2026-05-22，阶段 C 4 个 SHADOW 候选已经有**最小业务切片 + smoke evidence 入口**，但还没有完成 5 seeds × paper-suite-small 的真实 evidence，也没有进入阶段 D 决策：
  - **SYS-1（CPD 涌现 β_t 切换）**：已新增 `CPDSwitchReadout`，发布 PE spike / reward shift / switch recommendation 到 temporal consolidation readout；当前只读，不直接改 live β_t。
  - **COG-3（persona / regime geometry 漂移监控）**：已新增 evaluation-side `persona_geometry_drift` / `persona_regime_geometry_alignment` readout；只读，不写 regime / substrate / temporal owner。
  - **COG-1 reframed（least-control + counterfactual credit）**：已新增 `CreditSnapshot.least_control_readout`，并让 evaluation mid layer 抽取 least-control / counterfactual readout。
  - **COG-2 reframed（ToM / 多人归因）**：已新增 `ToMInterlocutorRecordCount` / `tom_record_counts_by_interlocutor(...)`，并让 dialogue `metric_means` 暴露 `tom_distinct_interlocutor_max` / `tom_record_total_max`。
  - **SHADOW profile + metric 面**：四条 profile 已注册并可通过 `default_phase2_shadow_evidence_profiles()` / `run_phase2_shadow_evidence_smoke.py` 显式跑；默认 ablation / strong-proof 矩阵不包含它们。
  - **阶段 D 决策机制**：`build_phase2_shadow_decision_report.py` 已能基于 multi-seed JSON 输出 `ACTIVE_CANDIDATE` / `REMAIN_SHADOW` / `DISABLED` 建议；真实 5-seed single-profile evidence 已跑完，四条单项 profile 均为 `REMAIN_SHADOW`（head-to-head winrate = 0.5）。当前缺 Phase 3 combo 真实 evidence 与 OA-4 full audit-agent，因此仍不是自动合并门。
  - **Phase 3 combo 路径 READY-TO-RUN（2026-06-10 冒烟核验）**：(a) synthetic 全
    profile（4 单项 + 3 combo + audit-persona-geometry）冒烟通过，artifact
    `artifacts/phase2-shadow-combo-smoke-synthetic`（manifest verified）；(b)
    **真实 runner** 单 seed `--case-limit 1 --seeds 0 --include-phase3-combos`
    端到端跑通（约 23 分钟含 HF 权重加载），artifact
    `artifacts/phase3-shadow-combo-smoke-singleseed`（manifest verified）。
    完整 evidence run 只差算力与择机执行：
    `python scripts/run_phase2_shadow_evidence_multiseed.py --case-limit 4 --seeds 0 1 2 3 4 --include-phase3-combos --output-dir artifacts/phase3-shadow-real-multiseed`
    （随后 `build_phase2_shadow_decision_report.py` + manifest verify）。结果
    无论方向，须如实回填本条 + deploy 侧 `VolvenceDeploy/docs/known-debts.md`
    `D-thesis-1`。
- **违反**：不违反 R 铁律。每个候选都遵循 Phase 1 的 profile composition + capability wiring 接口，行为隔离 + 可回滚 + 不污染现有 owner SSOT。COG-1 / COG-2 / COG-3 的"现状盲点"段落在 phase-a-brief 中已被 PARTIALLY-REFUTED，实际工作量比 [`探索方向.md`](moving%20forward/探索方向.md) 原描述小一档（4 个 ToM slot 已 ACTIVE / COCOA Phase 1.A+2.A 已上线 / multi_party_scenarios 已有 fixture）。
- **风险**：
  - **短期低**：不做不影响 functionality，现有系统照常运行；阶段 A brief 已经记录所有候选的 ROI 与依赖。
  - **中期中**：[`探索方向.md`](moving%20forward/探索方向.md) 中 P0 优先级"最高 ROI 12 项"在 SYS-1 / COG-1 / COG-2 / COG-3 没跑过实测前都只是研究建议，无法判断哪些值得继续投入。
  - **长期中**：阶段 D 决策机制不建立 → 即使跑了 SHADOW evidence 也没有可重复的 ACTIVE 推进规则，每次靠人工判断容易引入 confirmation bias。
- **触发条件**：
  - **任何 P0 探索方向需要工程级证据**（投资人尽调 / 内部 evidence run / 学术 reference）→ 先运行 `python scripts/run_phase2_shadow_evidence_smoke.py --synthetic-runner` 检查 schema，再运行真实 runner 版本收集初版 evidence
  - **EQ / 关系质量评估**真想要可量化产出 → 启动 COG-2（多人场景 + ToM owner 拆分）
  - **反事实信用归因**真想要 long-horizon 解释 → 启动 COG-1（least_control + commitment lineage）
  - **ModificationGate 进入 rare-heavy artifact 路径** → 触发阶段 D 全套
- **推荐修法**（按 [`experiment.md`](moving%20forward/experiment.md) §6 推荐起跑顺序）：
  1. **已完成最小切片**：SYS-1 / COG-3 / COG-1 / COG-2 readout + profile + metrics + synthetic smoke。
  2. **下一步**：真实 runner 跑 `scripts/run_phase2_shadow_evidence_smoke.py --case-limit 1`，确认非 synthetic path 可完成；再扩到全 4 canonical cases。
  3. **已完成 wrapper / decision-support**：`run_phase2_shadow_evidence_multiseed.py` 与 `build_phase2_shadow_decision_report.py` 已落地；synthetic 2-seed + Phase 3 combo smoke 已跑通。
  4. **已完成真实 5-seed 单项 evidence**：`artifacts/phase2-shadow-real-multiseed` 已产出并通过 manifest 校验；decision report 四条均为 `REMAIN_SHADOW`，所以当前不建议切任何 behavior ACTIVE。
  5. **下一步**：真实 multi-seed 跑 Phase 3 combination profiles（`--include-phase3-combos`），验证组合是否有正 delta 或负迁移。
  6. **阶段 D — 组合 profile + ACTIVE 决策**：只有当真实 multi-seed + audit evidence 达到 `ACTIVE_CANDIDATE`，才进入人工 review + rollback-window；否则保持 SHADOW / DISABLED。
  6. **配套**：每条候选的 SHADOW evidence 走 [`docs/specs/<candidate>-shadow-evidence-<date>.md`](specs/) 模板沉淀，让 PR review 阶段一眼看到 metric_means delta + 何时切 ACTIVE 的判定基准；模板已由 B4 [`run_shadow_evidence_template.py`](../scripts/run_shadow_evidence_template.py) 自动生成。
- **优先级**：**低**（用户 2026-05-13 明确指示）。阶段 A brief 已经把每个候选的工作量 / 耦合 / 起跑前置都核查清楚，按条触发即可；不存在"必须在 X 时间前跑完"的硬截止。

---

# 商业化反思债（#45-#70）— 2026-05-13 一次性导入

> 来源：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 / §4.1-4.5 / §6 / §8 反向回查。三个最 promising 方向（P5 / P1 / P2）在商业层做出的承诺 vs 仓库工程层 evidence 的 gap。
> 编号约定：#45-#51 = 横切（三方向共依赖）；#52-#57 = P5 Companion Bench 专属；#58-#63 = P1 Figure-as-a-Service 专属；#64-#70 = P2 Growth-Advisor 专属。

---

## 45. 生产并发 / 多租户下的 latency / 显存 / 调度实测床缺失

- **路径**：
  - 缺位的目录：`tests/perf/`（不存在）+ `scripts/realistic_load_*.py`（不存在）
  - 推荐落点：[`packages/lifeform-service/`](../packages/lifeform-service/) + [`packages/dlaas-platform-launcher/`](../packages/dlaas-platform-launcher/) + [`packages/vz-substrate/`](../packages/vz-substrate/) 跨包共测
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §6.1 (单 ai_id × 月成本估算) / §8.1.1 (多 vertical 共载 latency 风险标"中-高 × 高")
- **问题**：[`docs/moving forward/summary.md`](moving%20forward/summary.md) §3 表已明确标注"没有看到 latency / 显存 / 调度的实测数据"。三个商业方向的报价 / SLO / 客户 demo 都隐式假设这一数据存在：
  - 单 substrate (Qwen2.5-32B / Llama-3.1-70B) 节点上，并发挂多少个 `LifeformSession` 时 P99 latency 还能 < 3s？没数据
  - 多 vertical 共载（5 个 vertical × N owner 总线 snapshot propagate）在并发下是否 contention？没数据
  - thinking loop / followup / tick 这些"用户没说话也跑"的内部 turn 在大量 ai_id 时占用多少推理预算？没数据
  - `PersonaLoRAPool.activate(...)` 多 session 同时切换不同 figure LoRA 的串行化代价？没数据
- **违反**：不违反 R 铁律。是工程基线缺失，不是契约违反。
- **风险**：**高**。三个商业方向都依赖这个不存在的数据集；第一次真实付费客户上量时同时点亮 P1 Figure SLO + P2 Growth-Advisor 多席位 + P5 Companion Bench 大并发跑分三个子问题
- **触发条件**：(a) 第一个 P1 / P2 客户的 PoC 进入并发测试阶段；(b) P5 公开榜单跑 120 scenarios × 头部 6 模型 × 多 SUT 的实际成本 / 时长估算需要；(c) DLaaS 灯塔客户 PoC 提出 SLO（uptime / latency / multi-tenant data residency）
- **推荐修法**：
  1. 新建 `tests/perf/` 目录 + `tests/perf/test_concurrent_lifeform_sessions.py` + `tests/perf/test_multi_vertical_owner_propagation.py` + `tests/perf/test_persona_lora_hot_swap_concurrency.py` 三个套件
  2. 新建 `scripts/realistic_load_companion.py` / `scripts/realistic_load_figure.py` / `scripts/realistic_load_growth_advisor.py` 各一个长跑脚本（`asyncio.gather` 并发 N 个 `LifeformSession`，按真实 turn 时间分布跑 30 min），输出 `artifacts/perf/<scenario>-<date>.json`（P50 / P90 / P99 latency / GPU mem peak / owner snapshot dispatch ms）
  3. 加 `docs/specs/perf-baseline.md` spec 落档当前 baseline + 三方向 SLO 拍板表（如"P1 figure 单 ai_id P99 < 5s @ 10 concurrent""P2 席位 P95 < 3s @ 50 end-user concurrent"）
  4. 守 R8：perf 测试只 read snapshots，不写 owner；只 read substrate generate latency，不改任何模型权重
- **优先级**：**中-高**（Phase A 第 1-2 个月必做，是横切 #46 / #47 / #49 / #61 / #70 共同前置）

---

## 46. 多租户两层 scope_key（tenant × end_user）+ 客户级 admin / end-user 删除路径

- **路径**：
  - 当前单层：[`packages/volvence-zero/.../memory/`](../packages/) `UserIdentity.scope_key`（closed-alpha 是 `user_id == scope_key`）
  - closed-alpha 文档：[`docs/closed-alpha-api-service.md`](closed-alpha-api-service.md) 明确"跨用户隔离走 scoped memory，alpha 阶段 user_id == scope_key"
  - 推荐落点：[`packages/lifeform-service/`](../packages/lifeform-service/) + [`packages/dlaas-platform-registry/`](../packages/dlaas-platform-registry/) 加 `TenantIdentity` × `EndUserIdentity` 双层 schema
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §2.1 (Tier-1 Scoped memory + 删除路径) / §6.3 (P2 客户 = 多席位)
- **问题**：closed-alpha 是"1 用户 = 1 scope"单层模型。但 P1 / P2 商业承诺都是 B2B：
  - 1 个 P2 母婴客户 = N 个 end user，需要"客户级 admin 可看 aggregated 报表 / end user 可独立删自己 memory / 客户终止合同则全删"
  - 1 个 P1 博物馆客户可能挂多个 figure bundle (Einstein + Curie + Tesla)，需要按客户级 admin 视角看跨 figure 总用量
  - 当前 schema 把 tenant_id 和 end_user_id 当成同一层 → admin 只能用一个 user_id 操作；删除粒度只有"all or nothing"
- **违反**：不违反 R 铁律，但违反 R8 "owner 唯一所有"的精神（tenant 与 end_user 是两个 owner，被强行折叠到一层）
- **风险**：**中-高**。第一个 P2 客户上来就会问"我能看到我管理的 50 个 end user 的活跃度吗""我注销账号能不能一键全删"，答不上来直接停在 PoC
- **触发条件**：(a) 第一个 P2 / P1 私有部署客户进入合规审计阶段；(b) 任何客户提出"按 end user 行权删除"的 GDPR / 中国 PIPL 合规要求；(c) ops dashboard 想区分客户级 vs end-user 级视图
- **推荐修法**：
  1. 在 `lifeform-service` 引入 `TenantIdentity(tenant_id: str)` + `EndUserIdentity(tenant_id: str, end_user_id: str)` 双层；`UserIdentity.scope_key` 派生为 `f"{tenant_id}:{end_user_id}"`
  2. closed-alpha 单层路径自动派生为 `tenant_id == "alpha"` + `end_user_id == user_id`，向后兼容
  3. `DELETE /v1/users/me/memory` 增加 `DELETE /v1/tenants/{tid}/users/{uid}/memory` + `DELETE /v1/tenants/{tid}/memory`（admin 删全 tenant）
  4. ops dashboard 加 tenant-aware view（admin scope vs end-user scope）
  5. 加 `tests/contracts/test_two_layer_scope_isolation.py`：tenant A 的 end_user X 与 tenant B 的 end_user X 即使同名也完全隔离；tenant A admin 只能 enumerate 自己 tenant 的 end users
  6. 与 #69 P2 专属端用户隔离条目同步设计（#69 是 P2 应用层的 surface，本条是横切 schema）
- **优先级**：**中**（Phase A 后期 / Phase B 早期；与 #49 evidence 删除路径联动）

---

## 47. substrate compatibility fingerprint + 升级降级路径未规约

- **路径**：
  - figure bundle: [`packages/lifeform-domain-figure/src/lifeform_domain_figure/figure_artifact.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/figure_artifact.py)（当前无 `compatible_substrates` 字段）
  - growth-advisor profile: [`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py`](../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py)（无 substrate fingerprint 字段）
  - companion-bench scenario: [`packages/companion-bench/src/companion_bench/spec.py`](../packages/companion-bench/src/companion_bench/spec.py)（scenario_hash 不含 substrate context）
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §8.1.1 直接列"substrate 升级对 figure bundle 兼容性破坏"为高 × 高风险，应对写"把 substrate compatibility fingerprint 写进 bundle；至少支持 N 和 N-1 两代"——但**只是策略表述，没落到 schema**
- **问题**：substrate 升级（Qwen2.5 → Qwen3 / Llama-3 → Llama-4）会让三方向以不同方式失效：
  - **P1 figure**：bundle.lora 直接绑死 substrate（PEFT 训练在某 substrate 上跑），换底座必须重新编译；steering vector 在新 substrate 上方向意义可能漂移；retrieval index / coverage map / refuser 应该 substrate-agnostic 但需要测
  - **P2 growth-advisor**：reviewed profile 编译进 application owner，理论上 substrate-agnostic；但 4 drives 通过 PE 影响 regime 在新 substrate 上的实际触发频率可能变化
  - **P5 companion-bench**：scenario_hash 不变但 SUT 用了新底座，老榜单是否需要重跑或加注？跨 substrate 升级的可比性如何保证
- **违反**：违反 R15 byte-level 回滚契约的隐含意图（"任何输入变化产生不同 artifact_id"——substrate 是隐式输入）
- **风险**：**中**。短期 substrate 不动就不出事；substrate 升级是必然事件（产品方至少每年一次）
- **触发条件**：(a) 任何 substrate 升级公告；(b) 客户问"换 substrate 是否要重新付编译费"；(c) #19 真材料 corpus 完成后第一次想跨 substrate 比较 bundle 行为
- **推荐修法**：
  1. 在 `vz-substrate` 加 `SubstrateFingerprint(model_id: str, version: str, weights_sha256: str)` typed dataclass + `OpenWeightResidualRuntime.fingerprint() -> SubstrateFingerprint`
  2. `FigureArtifactBundle` 加 `compatible_substrates: tuple[SubstrateFingerprint, ...]`（必填非空）+ `compute_bundle_integrity_hash` 折入；bake 时记录主 substrate，runtime activate 时检查兼容性 mismatch fail-loud
  3. `GrowthAdvisorProfile` 加 `validated_substrates: tuple[SubstrateFingerprint, ...]`（可空，空表示"通用"），runtime warn-if-mismatch
  4. companion-bench `RunRecord` schema 加 `sut_substrate_fingerprint`，公开榜单按 substrate 分组展示
  5. 加 `docs/specs/substrate-upgrade-protocol.md`：N+1 substrate 上线时，N bundles 的"必须重 bake / 可降级运行 / 完全不兼容"三档判定流程
  6. 加 `tests/contracts/test_substrate_fingerprint_propagation.py` 守门
- **优先级**：**中**（Phase A 后期 / Phase B 早期；substrate 升级是滞后但必然事件）

---

## 48. LLM-as-judge / LLM-as-classifier 的 self-preference 与跨家族方差量化

- **路径**：
  - P5 judges：[`packages/companion-bench/src/companion_bench/judge_perturn.py`](../packages/companion-bench/src/companion_bench/judge_perturn.py) + [`packages/companion-bench/src/companion_bench/judge_arc.py`](../packages/companion-bench/src/companion_bench/judge_arc.py)（spec §5 只规定"arc judge 来自不同模型家族"，无量化）
  - P2 archetype 识别：[`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py`](../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py) 5 archetype 定义，**识别机制空白**（见 #66）
  - P1 figure 4-gate：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/) 当前是 deterministic scoring（debt #41 closure 说明），但 verdict 升级路径如果引入 LLM judge 同样落入此坑
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §3.3 (P5 公信力)
- **问题**：MT-Bench / Chatbot Arena 早被诟病 LLM-judge 的 self-preference bias（GPT 当 judge 偏向 GPT 风格输出）+ 跨家族 inter-rater agreement 低。三方向都暗藏依赖：
  - **P5**：第一份榜单上线时如果没附 inter-judge κ 一致性 + 跨家族 SUT 排名方差，海外学术界会立刻质疑"VZ 的 judge 偏 GPT 所以 GPT 排第一"
  - **P2**：5 archetype 识别如果走 LLM classifier，不同 LLM family 给出不同分类 → 同一用户在不同时间被打不同标 → boundary policy 触发不一致
  - **P1**：未来如果用 LLM judge 替代 deterministic voice/cognition score，同样问题
- **违反**：不违反 R 铁律，但违反 R-PE / OA-1 "evaluation 是 readout 不是学习源"在**信息含量**层面的精神
- **风险**：**中-高**。P5 公开化的第一击就要面对这个质疑
- **触发条件**：(a) P5 第一份公开榜单准备发布前；(b) P2 第一个客户问"5 archetype 识别背后是什么模型"；(c) 任何 LLM judge 输出被引用为商业 SLO 指标
- **推荐修法**：
  1. 在 `companion-bench` 加 `scripts/companion_bench/judge_robustness_sweep.py`：用 N 个不同家族 LLM (GPT-5 / Claude Opus 4.7 / Qwen-Max / DeepSeek / Gemini) 当 per-turn judge × 同一组 SUT 输出，计算 inter-rater Spearman / Kendall κ + per-axis variance
  2. companion-bench `RunRecord` 加 `judge_robustness_summary` 字段（reference 跑分时附）
  3. 落 `docs/external/companion-bench-judge-robustness-v0.md` 公开报告（与 RFC §5 同步）
  4. P5 公开榜单 site 在每行 SUT 旁加 "judge variance σ" 列
  5. P2 / P1 任何引入 LLM classifier / judge 时 mandate 走同样的 robustness sweep；加 `tests/contracts/test_llm_classifier_robustness_required.py` AST 守门（任何新 `LLMRubricGrader` / `LLMClassifier` 必须有伴生的 robustness manifest）
- **优先级**：**中-高**（Phase A 必做，与 #52 同时跑；P5 公开化第一击的可信度地基）

---

## 49. evidence_root_dir 的可删除性（PIPL / GDPR）路径未明确

- **路径**：
  - 当前实现：[`docs/closed-alpha-api-service.md`](closed-alpha-api-service.md) §"DELETE /v1/users/me/memory" 已实现 scoped memory 的 DELETE，但只删 memory store，不删 `evidence_root_dir/sessions/*.json`
  - 推荐落点：[`packages/lifeform-service/`](../packages/lifeform-service/) `alpha.AlphaServiceConfig` + 新加 `evidence_deletion_policy` 字段
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 ("可被合规观察 + 用户可被遗忘") / §8.1.4 (法律/合规风险表 "用户被遗忘权" 标"中 × 中"应对仅"已实现 + minimal scope")
- **问题**：当前 evidence 文件是 audit 资产但同时也是**个人信息**：
  - P1 博物馆客户：用户在 demo 上的对话 evidence，能不能按用户 ID 删？
  - P2 私域客户：end user 行权要求删除，evidence 怎么删？删了之后 audit 完整性怎么 reconcile？需要"删除证据"留 placeholder
  - P5 公开 transcript（提交方授权）vs 私有 held-out transcript（含 simulator 输出），两者留存策略不同
  - 当前 closed-alpha-api-service 是 minimal scope，没有"evidence 可按用户删 + 留下删除证据"路径
- **违反**：不违反 R 铁律，但 P1 / P2 / P4 进任何合规要求高的客户尽调时必查项
- **风险**：**中-高**。法律风险——一旦客户行权而 VZ 答不上"evidence 删除路径"就触发合规事件
- **触发条件**：(a) 第一个 P1 / P2 客户合规审计；(b) 任何 end user 行权要求；(c) 监管约谈 / GDPR / PIPL inquiry
- **推荐修法**：
  1. 加 `lifeform-service.evidence.EvidenceDeletionPolicy(retention_days, delete_on_user_request: bool, retain_deletion_proof: bool)` typed config
  2. 加 `DELETE /v1/users/me/evidence?since=<iso>&until=<iso>` 端点；删除时把删除目标的 SHA-256 + scope_key + timestamp 写到 `evidence_deletion_ledger.jsonl`（append-only，永不删）
  3. `DELETE /v1/users/me/memory` 增加 `--include-evidence=true` 参数；旧调用默认行为不变
  4. P2 admin scope 加 `DELETE /v1/tenants/{tid}/users/{uid}/evidence`（与 #46 双层 scope 联动）
  5. 加 `docs/specs/evidence-deletion-protocol.md` 明确 "audit 需要 vs 用户行权" 的张力如何在 deletion ledger 上调和
  6. 加 `tests/contracts/test_evidence_deletion_proof_chain.py`：删除后 audit log 仍能 enumerate 删除事件 + scope_key + timestamp + sha256（但不能复原内容）
- **优先级**：**中**（Phase A 后期 / Phase B 早期；与 #46 双层 scope 同 packet 设计更高效）

---

## 50. Rollback drill 是 contract test 还是真生产实战未分清

- **路径**：
  - figure rollback：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py) `cmd_rollback` + audit append-only（debt #23 closure 文档）
  - PersonaLoRAPool rollback：[`packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py`](../packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py) context-manager 退出回滚（debt #20 closure）
  - learned-baseline rollback drill：[`tests/contracts/test_learned_baseline_rollback_drill.py`](../tests/contracts/test_learned_baseline_rollback_drill.py) 6 个 unit-test 级 drill（Wave E3 落地）
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 ("可回滚自修改门控") / §2.3 (R10 / R15) / §8.1.5 团队风险 "工程纪律和商业 KPI 之间产生张力"
- **问题**：`figure-vertical.md` 与多个 closure 文档都提到 `is_reversible=True + rollback drill 测试覆盖`。但当前 drill 是：
  - unit / integration test 里的合成场景，**不是**真在生产 substrate runtime 上跑 byte-identical revert 的实战测试
  - PersonaLoRAPool 的 byte-identical 退出是 hf-marked test 在 tiny-gpt2 上验证，不是真 Qwen / Llama 生产负载
  - cross-process / cross-restart 的 rollback 路径完全没测（重启服务后 audit log 还在，但内存 pool 重建了，rollback 链是否完整？）
- **违反**：不违反 R 铁律，但 R15 "可回滚 + counterfactual evidence" 在生产环境是否真站得住未验证
- **风险**：**低-中**。短期合成测试足够；长期第一次客户在生产上要求回滚时是头一次实战 → 风险事故
- **触发条件**：(a) 第一个 P1 客户合同写"提供回滚证据" SLA；(b) ModificationGate OFFLINE artifact 真上 ACTIVE；(c) 任何 substrate 升级后想回滚到 N-1
- **推荐修法**：
  1. 加 `tests/perf/test_production_rollback_drill.py`（依赖 #45 perf 床）：真在 Qwen 1.5B+ 上加载 figure bundle → 10 turn 真生成 → 触发 rollback → 再 10 turn 验证 logits 与 base substrate byte-identical 等价
  2. 加 `scripts/rollback_drill_<vertical>.sh` 一键脚本；纳入"每月生产验证"运维节奏
  3. cross-restart 路径：`bundle.pickle` reload 后 integrity_hash 必须与原 byte-equal（debt #23 closure 已守门 unit 级，加 service 级 reload→activate→rollback 完整链测试）
  4. 加 `docs/specs/rollback-drill-cadence.md` 明确 "每月" / "每次 substrate 升级前" / "每个新 figure bundle 上线前" 三档
- **优先级**：**低-中**（Phase B 中期；P1 / P4 商业承诺触发时拉起）

---

## 51. "关系连续性"——所有差异化卖点的根，但没有真实可测 ground truth

- **路径**：
  - 当前 readout：rupture/repair count + companion-bench A3 (callback recall LLM judge) + il_rapport / bond_warmth (vitals readout)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 ("关系连续性 + 可治理性 + 多角色复用" 三件组合包) / §3.3 (差异化定位) / §2.1 (Tier-1 资产)
  - 上游警告：[`docs/moving forward/summary.md`](moving%20forward/summary.md) §2.5 ("评估自身可信度问题——递归陷阱")
- **问题**：VZ 卖的核心差异化是"关系连续性"。但**关系连续性怎么测量**？
  - rupture/repair count 是**系统自评**（系统说自己 rupture 了 X 次），不是"用户感受被理解"
  - companion-bench A3 是 LLM judge 判断"模型是否记住"，不是"用户是否觉得被记住"
  - P2 月报里"用户活跃度 / boundary 触发率"是 proxy，不是 LTV 因果
  - 仓库里**没有任何**"系统自评指标 vs 用户真实感受"的对照实验数据
- **违反**：不违反 R 铁律。R12 evaluation 是 readout 不是学习源——但 readout 的可信度本身缺 external validity 验证
- **风险**：**中-高**。这是 §8.1.5 "评估自身可信度"在商业层的具体落地；30 天 P2 试点结束后的月报数字如果没有外部对照 evidence，就是"我们觉得我们做得不错"
- **触发条件**：(a) 第一个 P2 30 天试点设计阶段；(b) P5 第一份榜单的 A3 子轴需要交叉验证；(c) 任何客户问"你怎么证明用户真的觉得关系连续"
- **推荐修法**：
  1. Phase A P2 试点必须前置设计"双盲第三方评分"协议：把同一段 30-turn 对话片段打乱（VZ 输出 vs baseline LLM 输出 vs 真人客服输出），让招募的 N=20 评估员盲打 A3 类指标，与 system 自评对比
  2. companion-bench A3 加伴生"human eval cross-validation"轨道（与 debt #33 human-eval 联动）
  3. 加 `docs/specs/relationship-continuity-external-validation.md`：列举系统自评指标 vs 用户感受 proxy 的对照矩阵 + 评估方法论
  4. P2 月报 metric 旁标注 "system self-eval" vs "external-validated"，避免在客户面前混用
- **进展（2026-06-10）**：修法 1 + 3 + 4 的**规范半**已落地——
  [`docs/specs/relationship-continuity-external-validation.md`](specs/relationship-continuity-external-validation.md)
  固化了对照矩阵、双盲三臂评分协议（N=20 评估员 / 30-turn 片段 / VZ vs baseline vs
  真人）、两个判读门（效度门 ρ≥0.4 / 差异门配对显著 + 二选一胜率>0.6）与
  `system_self_eval / llm_judge / external_validated` 三态标注枚举（已登记
  `00_INDEX.md` §8）。**evidence run 本身仍 OPEN**（评估员招募 + 片段采集 + 真实
  评分未做）；部署侧 `VolvenceDeploy/docs/known-debts.md` `D-thesis-1` EXIT 条件 (b)
  以本 spec 的两个判读门为可执行定义。
- **优先级**：**低-中**（Phase B 中期；与 P2 30 天试点同 packet 设计；与 #33 human-eval 联动）

---

## 52. Companion Bench 6 轴权重 (0.10/0.15/0.25/0.20/0.10/0.20) + A6 cap=60 calibration 来源未落档

- **路径**：
  - 实现：[`packages/companion-bench/src/companion_bench/aggregator.py`](../packages/companion-bench/src/companion_bench/aggregator.py)（`A6_CAP_THRESHOLD = 60.0`、`A6_CAP_VALUE = 50.0`、6 轴权重 hardcoded）
  - spec：[`docs/specs/companion-bench.md`](specs/companion-bench.md) §6 列了权重和阈值，但**没列怎么算出来的**
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.5 (P5 概率 60-75% 是基于 v1.0 已就绪的判断) / §7.2 (P5 GTM 第一击就要发布榜单)
- **问题**：这两个参数直接决定排名：
  - 给 A3 (关系连续性) 25% 权重是工程直觉还是有 sensitivity analysis？
  - A6 cap = 60 阈值如果定低，安全较好的 SUT 会被错误封顶 50；定高，安全有问题的 SUT 排名虚高
  - 当前没附 robustness sweep（在 50/55/60/65/70 多个阈值下排名是否稳定）→ 头部 LLM 厂商提交后第一句质疑就是"为什么是 60 不是 55"
- **违反**：不违反 R 铁律，但违反"评估指标必须可解释 + 可证伪"的精神
- **风险**：**中-高**。P5 公开化的第一击就要面对这个质疑；如果第一份榜单出来后某厂商质疑且无法答辩，公信力直接崩
- **触发条件**：(a) P5 第一份公开榜单准备发布前；(b) RFC v0.1 → v1.0 升级 working group 公开评论期；(c) 任何头部厂商提交跑分后排名靠后并提质疑
- **推荐修法**：
  1. 加 `scripts/companion_bench/calibration_sweep.py`：6 轴权重各 ±0.05 范围 sweep + A6 cap 在 50/55/60/65/70 sweep；输出 reference 10 SUT 排名稳定性矩阵
  2. 加 `docs/external/companion-bench-calibration-report-v0.md`：公开 sweep 结果 + 当前权重选择的论证（"A3 权重 0.25 因为长程陪伴的核心价值轴" + 引用 EQ-Bench 3 / RP-Bench 等同行口径）
  3. RFC v1.0 升级时把 calibration report 作为 §6 必备附录
  4. Companion Bench site 加 "Why these weights?" page + interactive sweep widget（可选）
- **优先级**：**中-高**（Phase A 必做；P5 公开化前置）

---

## 53. Companion Bench user_simulator 的 bias 注入测量

- **路径**：
  - 实现：[`packages/companion-bench/src/companion_bench/user_simulator.py`](../packages/companion-bench/src/companion_bench/user_simulator.py)（LLM-backed user + deterministic FSM 16 actions）
  - spec：[`docs/specs/companion-bench.md`](specs/companion-bench.md) §4 列 16 actions，但未规定 simulator LLM 选择对 SUT 评分的影响
- **问题**：simulator 用 LLM 时会把自身偏好（如倾向 over-directive / 倾向 verbose）注入 SUT 测试。如果 simulator 用 GPT，GPT 当 SUT 时会显得"和 user 投契"——这是 P5 公正性的命门：
  - 当前 spec 没规定 simulator 必须来自和 SUT 不同家族
  - 没有"用 N 个不同家族 LLM 当 simulator × 同一 SUT × 同一 scenario，看 6 轴 final 分数方差"的实测
- **违反**：不违反 R 铁律，但与 #48 LLM-judge bias 同构
- **风险**：**低-中**。短期没人做 simulator 选择质疑；长期被严肃 reviewer 发现后 P5 中立性受损
- **触发条件**：(a) 头部模型厂商提交跑分后开始反向工程榜单方法论；(b) 学术 reviewer 在 RFC v1.0 公开评论期质疑；(c) #48 robustness sweep 触发时一并做
- **推荐修法**：
  1. 加 `scripts/companion_bench/simulator_robustness_sweep.py`：N 个家族 LLM 当 simulator × 固定 5 个 reference SUT × 24 公开 scenario，输出 per-SUT × per-axis variance
  2. spec §4.x 新加 "Simulator family rotation" 段：明确每季度公开榜单跑分时，simulator LLM 必须从公布的 4+ 家族池随机抽
  3. companion-bench `RunRecord.simulator_family` 必填字段；榜单 site 显示 simulator family
- **优先级**：**低**（Phase B 中期；与 #35 季度治理自动化联动）

---

## 54. Companion Bench 120 scenario × 6 轴的 statistical power 未量化

- **路径**：
  - 公开 24 scenario：[`packages/companion-bench/src/companion_bench/scenarios/public/`](../packages/companion-bench/src/companion_bench/scenarios/public/)
  - 私有 96 held-out：`external/companion-bench-heldout/`（git submodule）
  - 缺位：`scripts/companion_bench/statistical_power_analysis.py`（不存在）
- **问题**：24 公开 + 96 私有 = 120 场景。两个 SUT 排名差多少 ELO 才算"显著"？没有 power analysis：
  - noise floor 高 → 每次新提交都让排名乱跳，公信力崩
  - noise floor 低 → 可以让小差异稳定排名，可信度强
- **违反**：不违反 R 铁律
- **风险**：**低-中**。第一份榜单跑出来后稳不稳全靠运气
- **触发条件**：(a) #52 calibration sweep 触发时一并做；(b) 第二份榜单更新时如果排名大幅波动；(c) RFC v1.0 公开评论期被问"why 120 not 240"
- **推荐修法**：
  1. 加 `scripts/companion_bench/statistical_power_analysis.py`：固定 5 reference SUT × 多次 seed 重跑 × 计算 per-axis ELO 95% CI
  2. 加 `docs/external/companion-bench-statistical-power-v0.md` 公开报告
  3. 榜单 site 每行 SUT 加 "ELO ± 95% CI" 而不是单一数字
  4. 如果 power 不足，触发 v1.x 扩 scenario 到 200+
- **优先级**：**低**（Phase B 中期；与 #52 一同做更高效）

---

## 55. Companion Bench 跨语言 scenario 平衡（中文 / 英文）

- **路径**：
  - 公开 scenarios：[`packages/companion-bench/src/companion_bench/scenarios/public/`](../packages/companion-bench/src/companion_bench/scenarios/public/)（24 个，需查中英文占比）
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §7.7 (国际化与中文市场取舍：P5 优先英文市场启动)
- **问题**：P5 GTM 优先英文市场（学术影响力 / arXiv），但 24 公开 scenario 当前是中文为主还是双语未明确：
  - 如果第一份榜单跑出来全是中文场景，海外 LLM 厂商有理由说"你 benchmark 偏中文场景，我家英文模型当然吃亏"
  - 没有 "scenario.language" 字段 + 跨语言子榜单
- **违反**：不违反 R 铁律
- **风险**：**低-中**。海外 GTM 启动时被质疑可比性
- **触发条件**：(a) P5 准备投 arXiv preprint + 英文媒体 PR 前；(b) 第一个海外学术合作团队（Stanford CRFM / 港中文）询问；(c) 海外厂商提交跑分后质疑
- **推荐修法**：
  1. `ScenarioSpec` 加 `language: Literal["zh", "en", "bilingual"]` 字段（必填）
  2. 公开 24 scenario 至少凑够 12 中 + 12 英平衡（如不足，补充翻译或新增）
  3. private held-out submodule 同样 48 中 + 48 英平衡
  4. 榜单 site 加跨语言子榜单 view（中文 / 英文 / 综合）
  5. spec §3 / RFC §3 同步更新
- **优先级**：**低-中**（Phase A 后期；P5 英文市场启动前置）

---

## 56. Companion Bench 持续每季度更新的成本闭环未精算

- **路径**：
  - 跑分入口：[`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py)
  - cost 模块：[`packages/companion-bench/src/companion_bench/cost.py`](../packages/companion-bench/src/companion_bench/cost.py)（`CostTracker` 已实现，能算单次 run 成本）
  - 缺位：完整跑分总成本估算 + 季度更新预算闭环
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §7.2 (P5 GTM 成本"头部模型 API 调用费 10-30 万")
- **问题**：需要精确算出"单次完整跑分（120 场景 × 平均多少 turn × （SUT + per-turn judge + arc judge）× 头部模型单价）"的总成本：
  - 实算 > 50 万 / 季度，"每季度更新"就守不住，公信力随时间衰减
  - 当前 `_DEFAULT_PRICES` 是默认价格表，但没汇总跑过一次完整估算
- **违反**：不违反 R 铁律
- **风险**：**低**。商业承诺与运营预算的对账问题
- **触发条件**：(a) Phase B 季度更新预算审批；(b) 需要给融资 deck 一份"P5 持续运营成本"
- **推荐修法**：
  1. `scripts/companion_bench/estimate_quarterly_cost.py`：基于 `CostTracker` 已 record 的真实 token 用量 + 当前 default prices，模拟"10 reference SUT × 120 scenario × 8 季度"总成本
  2. 输出 `artifacts/companion_bench/quarterly_cost_estimate.md` 表
  3. 与 #34 staged executor 联动：成本超预算时优先跑公开 24 scenario，private held-out 季度跑 1 次
  4. 加 `docs/external/companion-bench-cost-model-v0.md` 公开成本模型（让提交方有预算预期）
- **优先级**：**低**（Phase B 中期）

---

## 57. Companion Bench 私有 held-out 的"trusted runner"机制未规约

- **路径**：
  - 当前 submission：[`packages/companion-bench/src/companion_bench/submission.py`](../packages/companion-bench/src/companion_bench/submission.py) + [`packages/companion-bench/src/companion_bench/heldout_loader.py`](../packages/companion-bench/src/companion_bench/heldout_loader.py)
  - submission protocol：[`docs/external/companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §10.2 反目标 "公开 held-out 提示集 → benchmark 价值归零"
- **问题**：submodule 是 git private，但提交方跑分时 SUT 公司一定会看到 prompt（除非走 trusted runner——VZ 替对方跑，对方只交模型 endpoint）：
  - 如果他们能直接看到 held-out scenario，submodule git private 等于摆设（一旦泄露价值归零）
  - 如果走 trusted runner，VZ 要扛对方模型的调用成本和接入成本
  - 当前 `submission.py + heldout_loader.py` 没规定"提交方是否能直接看到 held-out scenario"——这是 P5 商业模式的隐藏分支
- **违反**：违反 §10.2 反目标的精神（公开 held-out → benchmark 价值归零）
- **风险**：**中-高**。P5 公开化第二个季度就会被一个不诚实的提交方触发
- **触发条件**：(a) 第一个外部团队提交 held-out 跑分；(b) 任何头部厂商正式接入 submission queue
- **推荐修法**：
  1. 加 `docs/external/companion-bench-trusted-runner-protocol.md`：明确两种提交模式
     - "self-hosted run": 提交方自跑，但只能用公开 24 scenario，结果排在公开榜
     - "trusted-runner run": VZ 跑，提交方提供 OpenAI-compat endpoint + token，结果可上 held-out 完整榜
  2. 加 `scripts/companion_bench/trusted_runner.py`：VZ 侧执行 + 加密存储提交方 endpoint credentials + 自动调用 + 跑完销毁 transcript（只留 verdict）
  3. 加 `docs/external/companion-bench-heldout-leak-protocol.md`：泄露事件应对（rotate held-out / 取消榜单 / 公告）
  4. 加 `tests/contracts/test_heldout_access_audit.py`：任何 read held-out scenario 的 code path 必须经过 audit logger
- **优先级**：**低-中**（Phase B 中期；第一个外部 held-out 提交触发）

---

## 58. P1 L4 ScopeRefuser 的 false refuse / false answer 双向准确率 ground truth 缺失

- **路径**：
  - 实现：[`packages/lifeform-expression/src/lifeform_expression/scope_refuser.py`](../packages/lifeform-expression/src/lifeform_expression/scope_refuser.py)
  - 验证管线：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/) (Wave O-P + 5 道 reviewer-curated OUT_OF_SCOPE_REFUSAL_QUESTIONS)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 ("拒答它没被授权说的话") / §4.1 (P1 价值主张 "L3 引证 + L4 拒答让博物馆/教育机构的法务可以签字")
  - 关联：debt #39 (Wave K bundle coverage_map 过严) + #42 (5 道样本 refusal-precision 离散度过粗)
- **问题**：L4 是 figure 路径的法律生死线。但 spec 没有：
  - false refuse rate 的可接受上限（用户问了实际有 corpus 覆盖的话题，被错误拒答）
  - false answer rate 的可接受上限（应拒未拒，让 LLM 用 substrate 先验冒充人物）
  - 用来量化这两个 rate 的 ground truth set（每个 figure 都需要一个 reviewed "in-scope vs out-of-scope" 测试集）
  - Wave O-P 的 4-gate verdict 的 refusal scoring 只用了 5 道 OUT_OF_SCOPE 探针 + reviewer 模板 in-corpus 题，**单 figure 单语料量级**
- **违反**：不违反 R 铁律，但 #42 已经标注 5 道 OOS 样本 80% threshold 在小样本上离散度过粗
- **风险**：**中-高**。第二个 figure 上线就是裸奔（如果只有 Einstein 有完整 GT set）；P1 客户尽调时挑 10 段 demo 手工核查任何一段拒答错误都会让法务驳回
- **触发条件**：(a) 第二个 P1 figure 准备上线；(b) 第一个博物馆客户合同进入法务审查；(c) 任何"误拒可接受率" SLA 写进合同
- **推荐修法**：
  1. 每个 figure bundle 必须配套 `data/figure_refusal_gt/<figure_id>/in_scope.jsonl`（≥ 50 题，reviewer 标注 "expect_answer + cited_chunk_ids"）+ `out_of_scope.jsonl`（≥ 50 题，reviewer 标注 "expect_refuse + reason"）
  2. 加 `scripts/figure_refusal_eval.py`：跑 GT set × 计算 false_refuse_rate / false_answer_rate / per-rate 95% CI
  3. `FigureArtifactBundle` 加 `refusal_eval_report` 字段（必填非空，bundle 编译时自动跑 GT set）
  4. P1 合同模板里 SLA 写"在 reviewed GT set 上 false_refuse_rate ≤ 0.1 + false_answer_rate ≤ 0.05"，而不是模糊承诺
  5. 加 `tests/contracts/test_figure_bundle_refusal_gt_required.py`：任何 production-tier bundle 必须 ship `refusal_eval_report` 非空
  6. 与 debt #42 (5 道样本→ 50 题) + #39 (coverage_map 过严的修正)联动
- **优先级**：**中-高**（Phase A 必做，P1 客户尽调直接命门）

---

## 59. P1 L3 GroundedDecoder 引证 hallucination 检测率未量化

- **路径**：
  - 实现：[`packages/lifeform-expression/src/lifeform_expression/grounded_decoder.py`](../packages/lifeform-expression/src/lifeform_expression/grounded_decoder.py) `verify_with_pointers` 返回 typed `EvidencePointer`
  - cognition scoring：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/) cognition score = retrieval_index `assertion_is_supported` cosine
  - 关联：debt #41 (cognition gate 通过的硬前提=真 Qwen-1.5B PEFT bake)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 ("它说的每一句话都能溯源" 是 P1 第一卖点) / §2.4 (figure bundle artifact 切换成本)
- **问题**：GroundedDecoder.verify_with_pointers 返回 EvidencePointer。需要量化的指标：
  - pointer 指向的原文段落是否真的支持那段断言？（不是 keyword 匹配，是语义支持）
  - "未引证 + 用 substrate 先验生成"的实质性断言占比？
  - P1 客户尽调一定会做"挑 10 段 demo 的引证手工核查"——任何一段 hallucination 都会让法务驳回
- **违反**：不违反 R 铁律
- **风险**：**中-高**。法律生死线，与 #58 同级别
- **触发条件**：(a) 第一个 P1 客户合同进入法务审查；(b) 第二个 figure 上线；(c) 任何 demo PR 之前
- **推荐修法**：
  1. 加 `data/figure_grounding_gt/<figure_id>/assertions.jsonl`（≥ 100 题，每题含 question + expected_assertion + ground_truth_chunk_ids）
  2. 加 `scripts/figure_grounding_eval.py`：跑 GT set × 计算 evidence-faithfulness（pointer 真支持断言比例）+ unsupported-assertion-rate（无 pointer 但生成实质性断言比例）
  3. `FigureArtifactBundle` 加 `grounding_eval_report` 字段
  4. P1 合同 SLA 加"evidence_faithfulness ≥ 0.95 + unsupported_assertion_rate ≤ 0.05"
  5. 与 #41 真 Qwen PEFT 跑分联动（小 substrate 上 cognition score = 0 是噪音；真 Qwen 上 evidence faithfulness 才有意义）
- **优先级**：**中-高**（Phase A 必做，与 #58 同 packet）

---

## 60. P1 L1 StylePriorInjector 风格可感知性盲测

- **路径**：
  - 实现：[`packages/lifeform-expression/src/lifeform_expression/style_prior_injector.py`](../packages/lifeform-expression/src/lifeform_expression/style_prior_injector.py)
  - voice scoring：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/) voice = top80 overlap × 0.6 + sentence-length p50 match × 0.4（lexical proxy）
  - 关联：debt #40 (synthetic LoRA delta 经 LayerNorm 被吃掉，BUNDLE ≡ BUNDLE_LORA)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.1 (P1 价值主张 "听起来像他") / §6.2 (P1 单位经济假设 30 万首单)
- **问题**：`StylePriorInjector` 注入 hint tag，但底层 LLM 仍是 Qwen / Llama。"听起来像 Einstein"在普通用户盲测里是否能区分（vs 同样 corpus retrieval 但不加 style prior）？
  - 如果盲测区分度低，L1 不能作为独立卖点，必须 L1+L2 (LoRA) 组合卖
  - 这直接影响 P1 不带 OFFLINE-gate L2 的 minimum-viable 价格区间（30 万 vs 80 万的差异）
  - 当前 voice scoring 是 lexical proxy（top80 overlap + sentence-length），不是用户感知
- **违反**：不违反 R 铁律
- **风险**：**低-中**。短期 demo 用 reviewer 自审；长期定价拍板需要 evidence
- **触发条件**：(a) P1 第一份 pricing sheet 拍板；(b) 客户问"L1 vs L1+L2 我应该选哪档"；(c) #41 真 Qwen PEFT 完成后 voice gate 真量化
- **推荐修法**：
  1. 设计盲测 protocol：N=20 招募评估员 × M=30 段对话片段（混合 raw / bundle (L1+L3+L4) / bundle+LoRA (L1+L2+L3+L4) 三种 condition）× 5-point Likert "听起来像 X 的程度"
  2. 加 `scripts/figure_voice_blind_test.py` 半自动化 protocol（生成片段 / 打乱 / 收集打分 / 计算 Cronbach's α）
  3. `FigureArtifactBundle` 加 `voice_blind_test_report` 字段（首次盲测后填充）
  4. 与 #41 真 Qwen 跑分配套；P1 pricing sheet 引用真盲测 evidence
- **优先级**：**低**（Phase B 中期；P1 第二款 figure 上线时拉起）

---

## 61. P1 L2 LoRA hot-swap 在并发下的状态隔离 / 延迟实测

- **路径**：
  - 实现：[`packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py`](../packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py) + [`packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py`](../packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py) `TransformersOpenWeightResidualRuntime.activate_lora` (Wave D)
  - 测试：`packages/vz-substrate/tests/test_lora_aware_runtime_smoke.py` (debt #20 closure 7 case，含 `@pytest.mark.hf` 单 session 真 forward-hook，**不并发**)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §1.1 ("多角色复用 = 同一个内核服务多个垂直角色") / §8.1.1 (多 vertical 共载 latency 风险)
- **问题**：`PersonaLoRAPool.activate(figure_id, runtime=runtime)` Wave D 真改 forward-hook。需要实测：
  - 同时挂 N 个 figure LoRA 的 GPU 显存占用
  - session A 用 Einstein、session B 用 Curie，**严格并发**时 forward-hook 状态有没有 race condition？
  - LoRA 切换的 per-turn overhead（vs 不切换）
  - debt #20 closure 已守"嵌套抛 RuntimeError"——但只是单线程语义，不是真并发
- **违反**：不违反 R 铁律，但 R2 frozen base 在并发 forward-hook 切换下是否仍 byte-identical 未实测
- **风险**：**中**。第一次 P1 客户挂 ≥ 2 个 figure 同时跑就触发；同 §8.1.1 标"多 vertical 共载 latency 爆炸 = 中-高 × 高"
- **触发条件**：(a) 第一个 P1 客户挂多个 figure bundle；(b) #45 perf 床建好后第一次跑 figure 并发 stress；(c) substrate 升级到更大模型（Qwen 7B → 32B）
- **推荐修法**：
  1. 依赖 #45 perf 床；加 `tests/perf/test_persona_lora_concurrent_activation.py`：N=10 个 asyncio task 各 activate 不同 figure_id × 同时 forward 100 turn × 验证每 task 看到的 logits 与该 figure 单独 forward 一致
  2. 加 `scripts/realistic_load_figure_multi_persona.py`：真实模拟"10 用户 × 5 figure × 30 min"负载
  3. `LoRAAwareResidualRuntime` 加 thread-safety / asyncio-safety contract 注释；如发现 race 就加显式 `asyncio.Lock` per layer
  4. 加 `docs/specs/persona-lora-concurrency.md` 落档并发安全保证
- **优先级**：**低**（Phase B 中期；与 #45 同 packet）

---

## 62. P1 OFFLINE gate `validation_delta ≥ 0.05` 阈值的 measurement protocol 未固化

- **路径**：
  - gate 阈值：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/) `apply_steering_through_gate` / `apply_persona_lora_through_gate` (validation_delta ≥ 0.05)
  - kill criteria：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.1 P1 kill criteria "L2 steering / persona LoRA 真实开放权重底座效果 < 0.05 validation_delta → Bundle 编译价值被打折"
- **问题**：OFFLINE gate 卡 0.05，kill criteria 也用 < 0.05 砍 L2 价值。但**0.05 怎么测**？
  - 测哪个评估集？unit test 里的合成 prompt 还是真 in-scope corpus？
  - 测量 protocol 没固定 → 不同人测出来不同数 → gate 形同虚设
  - 当前 PEFT bake 的 `validation_delta = (init_loss − final_loss) / init_loss` 是训练 loss 改善，不是 downstream 行为改善
- **违反**：不违反 R 铁律，但违反 R12 evaluation 必须 verifiable / counterfactual 的精神
- **风险**：**低**。短期没人测；长期触发 kill criteria 时无法判定
- **触发条件**：(a) #41 真 Qwen PEFT bake 跑出真 validation_delta 时；(b) kill criteria 触发评估时；(c) ModificationGate audit 被外部 review
- **推荐修法**：
  1. 加 `docs/specs/figure-offline-gate-validation-protocol.md`：明确 validation_delta 测量协议（在 ≥ 50 题 in-scope GT × N 种 prompt formulation 上 of voice/cognition score 改善幅度，而不只是训练 loss）
  2. `apply_*_through_gate` 重构：proposal 里 mandate 同时给 train_loss_delta + downstream_score_delta，gate 看 downstream
  3. 与 #58 / #59 GT set 复用
- **优先级**：**低**（Phase B 中期；与 #41 真 Qwen 跑分配套）

---

## 63. P1 bundle 编译实测成本回填（替代估算 5-15 万）

- **路径**：
  - bake CLI：[`packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/cli/_commands.py)
  - 真采集证据：Wave K Einstein bundle (`figure-bundle:einstein:29eacd226a7cdfd0`) 6 SUCCESS / 5 cleaned / 2 reviewed
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §6.2 (P1 Bundle 编译 COGS 5-15 万估算) / §4.1 (P1 客单价 30-80 万)
- **问题**：§6.2 估"Bundle 编译 COGS 5-15 万"包含 GPU。Wave K 已经真跑了 Einstein bundle，**需要把这次的实际成本拆开**：
  - 工程师人天（含 reviewer human-in-loop 的时间）
  - L2 PEFT GPU 小时
  - corpus crawl 的 archive 访问 / rate limit 等待时间
  - 当前 §6.2 是估算锚点，但没真实数据回填 → 报价容易偏离
  - 如果 Einstein 实跑 > 30 万人民币，"30 万首单"就是亏本生意
- **违反**：不违反 R 铁律
- **风险**：**中**。第一个真付费客户报价时无 evidence 锚点
- **触发条件**：(a) P1 第一份正式 quote 准备发出前；(b) 第二个 figure（苏轼 / 居里夫人）准备启动；(c) 融资 deck 需要 P1 单位经济实证
- **推荐修法**：
  1. 加 `docs/business/figure-bake-cost-actuals.md`：把 Einstein Wave K 的实际工时 / GPU 小时 / archive 访问时间手工统计回填
  2. `FigureBakeAuditRecord` (debt #23 closure) 扩展 `cost_breakdown` 字段：bake CLI 自动记录起止时间、GPU 占用、reviewer 操作耗时
  3. 加 `scripts/figure_cost_summary.py`：从 audit log 出 per-figure cost 汇总
  4. P1 报价模板从估算改为"基于实测 N figure 平均"+ %CI
- **优先级**：**中**（Phase A 后期 / Phase B 早期；P1 第一份 quote 发出前）

---

## 64. P2 boundary policy 实际触发率分布的 baseline 缺失

- **路径**：
  - 实现：[`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py`](../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py) (4 boundary `bp-no-hard-sell` / `bp-no-overclaim` / `bp-no-flooding` / `bp-no-judgmental`)
  - boundary owner：vz-cognition / lifeform-expression boundary policy enforcer
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.2 P2 kill criteria "30 天 boundary 触发率 < 5% 或 > 50% 要重调架构" / §1.1 "拒答它没被授权说的话"
- **问题**：kill criteria 已经定下了 5% / 50% 双侧带，但**当前** `cheng_laoshi` profile 在合成测试集上的触发率分布是什么？
  - 如果合成测试都跑出 60%，30 天试点必然失败
  - 没有 baseline → 试点客户跑出来后无法判"是过严还是正常"
- **违反**：不违反 R 铁律
- **风险**：**中-高**。30 天试点失败的最快路径
- **触发条件**：(a) P2 第一个 30 天试点启动前；(b) `cheng_laoshi` profile 任何修改后；(c) 合成 benchmark 增加 boundary 相关 scenario
- **推荐修法**：
  1. 设计 `data/growth_advisor_boundary_eval/cheng_laoshi/scenarios.jsonl`：≥ 100 段合成对话片段，reviewer 标注每段"应该触发哪个 boundary 不应该触发哪个"
  2. 加 `scripts/growth_advisor_boundary_eval.py`：跑合成 scenarios × 输出 per-boundary 触发率 + per-boundary precision/recall vs reviewer 标注
  3. 落 `docs/specs/growth-advisor-boundary-baseline.md`：当前 cheng_laoshi 在合成 set 上的 boundary 触发率分布（健康范围 5%-50% 的位置）
  4. 与 companion-bench 联动：在 companion-bench 加 growth-advisor 专属 family 6 scenario（A6 boundary 子轴），让 boundary 行为也进 P5 榜单
  5. 30 天试点 SLA 写"per-boundary 触发率 ∈ [基线 ± 50%]"
- **优先级**：**中-高**（Phase A 必做，P2 30 天试点直接前置）

---

## ~~65. P2 `applicability_scope` day-counter 路由的真实生效证据~~ — DEPRECATED 2026-05-14

> **Deprecated（不再追踪）**：节奏分层（"用户处于第几阶段"）已由 `BehaviorProtocol.TemporalArc.progression_signals` 承接（PE-driven 关系阶段，不是 calendar 7 天硬切），不需要独立的 day-counter owner。
>
> 配套清理已完成：
> - 删除 `docs/specs/growth-advisor-day-counter.md`（spec 不再适用）
> - 删除 `tests/contracts/test_growth_advisor_day_routing.py`（contract 不再存在）
> - `cheng_laoshi.py` strategy_priors 的 `applicability_scope` 已移除 `growth_advisor:dayN` 字符串残留（保留 `funnel:*` / `regime:*` tags）
> - `fixture_uptake.py` 和 prose 文档（prd / archetecture / SYSTEM_DESIGN / SYSTEM_GUIDE / DATA_CONTRACT / commercialization-assessment / protocol-runtime / external-validation-protocol）相应注释/章节已修订
> - `growth-advisor-pilot-packet.md` G-B 子包整段下线，`commercialization-evidence-rollout.md` 周历表对应行删除
>
> 后续若仍需"用户处于关系阶段 X"信号，走 `BehaviorProtocol.TemporalArc.progression_signals`（PE-driven），由 protocol-runtime 模块在 application owner 中消费，不再回到 day-counter 路径。
>
> 历史背景保留以便审计：原 debt 描述见 `git log -- docs/known-debts.md` (commit history)。

---

## 66. P2 5 archetype 识别机制的明确选型

- **路径**：
  - 数据定义：[`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py`](../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py) `GrowthAdvisorKnowledgeSeed.domain == "user_archetype"` 5 类 mom 心态
  - 识别机制：**未实现**（profile.py 只列了 archetype 定义，没说怎么识别用户属于哪类）
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.2 P2 ("5 archetype × 7 day × 4 funnel × 4 boundary" 是 cheng_laoshi 的核心结构)
- **问题**：profile.py 列了 5 类 mom 心态（焦虑 / 对照 / 求标 / 吐槽 / 直接问产品）—— 但识别机制空白。三种可能：
  - (a) LLM classifier 每 turn 都跑 → 成本 / latency 上升（与 #48 LLM-judge bias 同坑）
  - (b) keyword/heuristic → **直接违反 `no-keyword-matching-hacks.mdc` 铁律**
  - (c) 学到的 metacontroller 切换单元 β_t → 符合 R3/R4，但**需要训练数据和 evidence**
  - 当前 (c) 没有，(b) 不能做，(a) 没决定 cost 模型
- **违反**：(b) 路径直接违反 `no-keyword-matching-hacks.mdc`；当前未选型 = 设计空白
- **风险**：**中**。P2 产品化前必须明确的技术设计 + 单位经济决策
- **触发条件**：(a) P2 第一个 30 天试点准备启动前；(b) cheng_laoshi profile 想真按 archetype 路由 boundary / playbook
- **推荐修法**：
  1. 落 `docs/specs/growth-advisor-archetype-detection.md`：评估 (a)/(c) 两路径
  2. **短期推 (a) LLM classifier**：在 vz-application 加 `ArchetypeClassifier` Protocol + `LLMArchetypeClassifier` 实现，每 N turn（不是每 turn）跑一次更新 archetype state；附 robustness sweep（与 #48 同 protocol）
  3. **长期看 (c)**：当 metacontroller (debt #44 顺位 1 SYS-1 CPD β_t 切换) 真上线后，archetype 作为 β_t emergence 的具体应用域
  4. 加 `tests/contracts/test_no_keyword_archetype_detection.py` AST 守门：禁止 archetype 识别代码包含 string-contains pattern
  5. 单位经济模型：把 archetype classifier 调用成本算进 §6.3 表
- **优先级**：**中**（Phase A 后期；P2 第一个试点前必须选型）

---

## 67. P2 月报 metric aggregation 的契约面缺失

- **路径**：
  - 当前：[`packages/lifeform-service/`](../packages/lifeform-service/) 已有 `weekly-report` 能力（closed-alpha 文档提及）
  - 缺位：月报 metric aggregation 的 contract / spec / owner
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.2 P2 ("月度可审计运营报告——audience 分析 + dialogue_external_outcome typed enum 自动化产出") / §7.4 GTM 第 4 条 "月报营销化—升级成给品牌总监看的月度运营报告"
- **问题**：P2 卖点"月报让我老板满意"。需要确认：
  - 月报每个 metric（rupture 数 / repair 率 / boundary 触发数 / archetype 分布 / 活跃度）从哪个 owner snapshot aggregate？
  - aggregation 逻辑是不是有 contract test 守门？schema 变更时月报历史可比性如何保证？
  - 月报本身是不是新的 owner / artifact？归谁所有？（避免破坏 R8）
  - 当前是已有能力但底层契约面尚未设计
- **违反**：违反 R8 owner 唯一所有（如果月报 aggregator 在多处实现就成了第二编排面）
- **风险**：**低-中**。短期可用 closed-alpha weekly-report；长期客户提出 schema 一致性需求时塌
- **触发条件**：(a) P2 第一个客户合同写"月报字段稳定性"；(b) 第二个 P2 客户上来后报表跨客户可比性需求；(c) #14 audience analysis 升真分析后想统一 pipeline
- **推荐修法**：
  1. 加 `docs/specs/growth-advisor-monthly-report.md`：明确月报 schema + 每个字段的 owner + aggregation 公式
  2. 在 lifeform-service 加 `MonthlyReportOwner` 模块：从下游 owner snapshot aggregate 出月报 typed dataclass，发布到月报专属 slot
  3. 加 `tests/contracts/test_monthly_report_schema_stability.py`：schema 变更必须显式版本号 + 历史报表读法
  4. 与 #14 audience analysis 共用 LLM provider；月报中"archetype 分布"调用 #66 archetype classifier 统计
  5. 月报 PDF / HTML 渲染层走 lifeform-expression（typed → 文本，符合 R4）
- **优先级**：**低-中**（Phase B 早期；P2 第一个客户续约前）

---

## 68. P2 GrowthAdvisorDrivePrior 4 drives 真生效的 ablation evidence

- **路径**：
  - 数据：[`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py`](../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profile.py) `GrowthAdvisorDrivePrior` (trust_building / empathy_response / restraint_against_pitch / kb_share)
  - 编译目标：`lifeform_core.DriveSpec`
  - 上游警告：[`docs/moving forward/summary.md`](moving%20forward/summary.md) §3 表 "PE-as-primary-signal 在开放对话上的可操作化失败" 是中风险
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.2 P2 ("反销售边界是合同条款不是 prompt")
- **问题**：4 drives 通过 PE 影响 regime 切换。具体到 P2：
  - 如何证明 `restraint_against_pitch_drive` 真的让 AI 在 day 3 不推销？（不只是它 compile 进了 owner，而是它的 PE 信号真改变了 regime）
  - 4 drives 之间的相互调谐怎么测？需要 ablation：去掉 `restraint_against_pitch_drive` 后行为变化是否符合预期？
  - 没有这组 ablation evidence，P2 卖"反销售边界是合同条款不是 prompt"就只是修辞
- **违反**：不违反 R 铁律，但违反 R-PE / R12 evaluation verifiable 精神
- **风险**：**低-中**。客户合规审计时质疑"你怎么证明 drive 真生效"
- **触发条件**：(a) P2 客户合规审计；(b) drive priors 修改后；(c) #44 SYS-1 CPD β_t 候选启动后想看 archetype 路由 + drive ablation 联动
- **推荐修法**：
  1. 加 `scripts/growth_advisor_drive_ablation.py`：4 个 condition (full / no-restraint / no-empathy / no-trust) × 同样 N 段对话 fixture × 输出 boundary 触发率 / regime 分布 / response style 对比
  2. 落 `docs/specs/growth-advisor-drive-ablation-evidence.md` 作为客户尽调材料
  3. 与 #64 boundary baseline 联动：ablation 结果应该和 baseline 配套展示
  4. 长期：当 #44 SYS-1 CPD β_t emerge 真起效后，drive 信号驱动 β_t 切换的因果链应有 evidence
- **优先级**：**低-中**（Phase B 中期；与 #64 同 packet）

---

## 69. P2 端用户两层 scope_key（tenant × end_user）— 应用层 surface

- **路径**：
  - 横切 schema：参见 #46（基础设施层）
  - P2 应用层缺位：[`packages/lifeform-service/`](../packages/lifeform-service/) growth-advisor 路径的 admin / end-user 两套 endpoint
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.2 P2 (席位制 = N 端用户) / §6.3 (10 席位 / 客户)
- **问题**：closed-alpha 是 single scope_key。P2 客户 = 1 个母婴品牌 → N 个 end user：
  - 需要 `(tenant_id, end_user_id)` 两层并行的 scope（基础设施层在 #46）
  - 这直接影响 ops dashboard 设计（admin 看 aggregated / end user 看 own）
  - 月报 (#67) 跨 end user 聚合需要 admin scope；GDPR 删除路径 (#49) 需要 end user scope
- **违反**：与 #46 同
- **风险**：**中**。第一个 P2 客户上来必查
- **触发条件**：(a) P2 第一个客户合同；(b) ops dashboard 设计阶段
- **推荐修法**：
  1. 等 #46 横切 schema 落地后，在 lifeform-service 加 `/v1/tenants/{tid}/admin/...` 一组 endpoint：list end users / aggregated metrics / bulk operations
  2. growth-advisor 月报路径走 admin scope 默认聚合 + end user scope 按需 drill-down
  3. handoff queue (#70) 按 tenant 隔离队列
  4. 加 `tests/service/test_growth_advisor_two_layer_scope.py`：tenant A 的 admin 看不到 tenant B 的 end users
- **优先级**：**中**（Phase A 后期 / Phase B 早期；与 #46 同 packet）

---

## 70. P2 handoff queue 在试点并发下的 SLO 实测

- **路径**：
  - 当前实现：closed-alpha 已有 handoff queue (`docs/business/commercialization-assessment.md` §2.1 Tier-1 资产)
  - 推荐位置：[`packages/dlaas-platform-ops/src/dlaas_platform_ops/`](../packages/dlaas-platform-ops/src/dlaas_platform_ops/)
  - 上游商业承诺：[`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §4.2 P2 ("合规：用户可删 / 处置可查 / 转人工可控")
- **问题**：closed-alpha 的 handoff queue 是 demo-级别。30 天试点的 1 个客户 + 10 席位 + 数百 end user，handoff 触发可能并发：
  - 队列容量 / 超时 fallback 行为？
  - 如果 SE 没及时接手，是 fail-safe 拒答还是降级回普通 LLM 回答？
  - handoff state 的持久化和跨重启恢复？
  - P2 第一个 30 天试点的 ops 团队只要遇到一次 handoff 队列丢失或超时未 fallback，就会怀疑"这套到底能不能上量"
- **违反**：不违反 R 铁律
- **风险**：**中**。试点客户感知差就直接砍续约
- **触发条件**：(a) P2 第一个 30 天试点 ops 团队接入；(b) handoff 触发频率超 demo 假设；(c) 服务重启 / 灾备演练
- **推荐修法**：
  1. 落 `docs/specs/handoff-queue-slo.md`：明确队列容量上限 / 超时阈值 / fallback 行为（推荐 STRICT_REFUSE 而不是降级回 LLM）
  2. 依赖 #45 perf 床；加 `tests/perf/test_handoff_queue_concurrent_load.py`：N 个 asyncio task 同时触发 handoff × 验证不丢 / 不串
  3. handoff state 持久化（与 evidence_root_dir 联动），重启后能 resume
  4. ops dashboard 加 handoff 队列 live view + alert（队列长度超阈值 / 超时未接手）
  5. 与 #69 两层 scope 联动：handoff 队列按 tenant 隔离
- **优先级**：**低-中**（Phase B 中期；P2 第一个试点 ops 接入前）

---

## 71. Companion Bench Qwen-only weak proxy judge 实测 6 axis 全 size-sensitive (强化 #48)

- **路径**：
  - 实测 driver: [`scripts/companion_bench/qwen_judge_robustness_replay.py`](../scripts/companion_bench/qwen_judge_robustness_replay.py)
  - 实测 artifact: `artifacts/companion_bench_smoke/judge_robustness_qwen_proxy.json` (gitignored)
  - 实测 narrative: `artifacts/companion_bench_smoke/SMOKE_REPORT.md` v2 §4
  - 现 smoke 默认 judge 配置: [`scripts/companion_bench/reference_systems.smoke_qwen.yaml`](../scripts/companion_bench/reference_systems.smoke_qwen.yaml)（per-turn=`qwen3-max`, arc=`qwen-plus`，同 family 不同 size）
  - 上游 spec：[`docs/specs/companion-bench.md`](specs/companion-bench.md) §5（"arc judge MUST come from a different model family than per-turn"）
  - 上游 debt：[`#48`](../docs/known-debts.md) (LLM-as-judge cross-family robustness sweep)
- **问题**：本次 smoke 用 4 个 lifeform-companion bundle × 3 个 Qwen judge (qwen3-max / qwen-plus / qwen-flash) replay 跑 arc-level scoring，得到：
  - **6 axis per-axis σ 全部 > 8（合格阈值）**：A1=32.68 / A2=26.04 / A3=9.52 / A4=32.07 / A5=25.42 / A6=13.43，平均 σ ≈ 23.2
  - **3 judge 给同一组 transcript 的 mean 分**：qwen3-max=40.83 / qwen-plus=34.38 / **qwen-flash=75.42** → 同 transcript 差 35 分
  - 单 SUT 没 pairwise，ranking flip = 0/0 (但绝对分数差 35 分本身就否定了 ranking 稳定性)
- **核心含义**：weak proxy（同 family 不同 size）**内部都不稳**——跨 family ρ 大概率更糟。这是 [#48](../docs/known-debts.md) 真 cross-family sweep **不是 nice-to-have，是 leaderboard 准入硬前提**的硬证据。
- **违反**：违反 [`docs/specs/companion-bench.md`](specs/companion-bench.md) §5 "arc judge 必须 cross-family" 精神；violations 不在 wheel 而在 orchestrator (`reference_systems.smoke_qwen.yaml` 配置同 family 两 model)，wheel 已显式不强制 family rotation。
- **风险**：**高**。**当前 smoke 跑分的所有绝对数字全部不可外引**——VZ 13.79 / Qwen 74.56 / ΔA3 都被 weak proxy bias 染色。任何用这组数据 cite 给客户/媒体/arxiv 都会被严肃 reviewer 一击即破。
- **触发条件**：(a) 任何 Qwen-only 配置进 leaderboard；(b) 任何 cite smoke 数据当 reference run；(c) [#32](../docs/known-debts.md) Companion Bench v1.0 launch 准备公开榜单时；(d) 任何 A/B prompt-engineering 实验依赖判分稳定性。
- **推荐修法**：
  1. **立即**：smoke `SMOKE_REPORT.md` 顶部加 `不可外引` watermark（已有 §4 标注 size-sensitive，但应升级为 doc-level 警告）
  2. **短期**：用户加 `OPENROUTER_API_KEY` 后切档 B (`--provider openrouter`，自动用 `openai/gpt-5-mini` per-turn + `anthropic/claude-3.7-sonnet` arc)，重跑同 4 scenario 验证是否 σ 显著下降（cross-family）
  3. **中期**：跑 [#48](../docs/known-debts.md) [`scripts/companion_bench/judge_robustness_sweep.py`](../scripts/companion_bench/judge_robustness_sweep.py) 真 cross-family sweep (5+ family × 5 SUT × 24 scenario)，validate Spearman ρ ≥ 0.75，作为 [#32](../docs/known-debts.md) launch 准入条件
  4. 长期：leaderboard `aggregate_results.json` 每行加 "judge_qualification_tier: A/B/C" 字段，weak proxy 跑分永不在 official tier 显示
- **优先级**：**中-高**（强化 [#48](../docs/known-debts.md) — #48 之前是"应该跑"，本次实测证明是"不跑就不能 launch"）

---

## 72. Smoke 默认 substrate=synthetic 让 VZ 跑 deterministic echo，触发 P5 kill criteria 误报警

- **路径**：
  - lifeform-serve CLI 默认: [`packages/lifeform-service/src/lifeform_service/cli.py`](../packages/lifeform-service/src/lifeform_service/cli.py)（`--substrate-mode {synthetic,hf-shared}`）
  - smoke runner 默认 substrate: [`scripts/companion_bench/run_companion_bench_smoke.py`](../scripts/companion_bench/run_companion_bench_smoke.py) line `VZ_SUT_SUBSTRATE = ... default "synthetic"`
  - smoke helper: [`scripts/companion_bench/start_vz_sut.sh`](../scripts/companion_bench/start_vz_sut.sh) `VZ_SUT_SUBSTRATE="${VZ_SUT_SUBSTRATE:-synthetic}"`
  - synthetic backend: `packages/vz-substrate/src/volvence_zero/substrate/adapter.py` `PlaceholderSubstrateAdapter` (deterministic echo / fake provider)
  - 上游商业承诺: [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §3.1 / §4.5 "P5 在长程陪伴 niche 上区分得出 SUT" 论点
- **问题**：smoke 跑 lifeform-companion SUT 默认走 `--substrate-mode synthetic`，底层 LLM 是 deterministic placeholder（不试图记忆 / 推理 / 共情，输出 `[echo:<sid>] <user_text>` 风格 placeholder）。本次 smoke 实测：
  - VZ-synthetic A3 Continuity = **5.00**（接近底）
  - Qwen3-Max v1 (real LLM) A3 = **36.25**
  - **ΔA3 = -31.25**，触发 [`commercialization-assessment.md`](business/commercialization-assessment.md) §4.5 P5 kill criteria 警告（"VZ A3 接近或低于 Qwen → 触发评估，需重审"）
  - 但这**完全不是 VZ architectural problem**——synthetic substrate 不试图记忆，是 substrate 选择的 unfair benchmark；任何 NLU 系统在 placeholder substrate 上都会 A3 = 0
- **核心含义**：smoke default substrate 选择**让数据看起来 unfair**，会让任何看到 SMOKE_REPORT 的人误以为"VZ 比 Qwen 弱"，污染商业判断。SMOKE_REPORT v2 §3 已加 synthetic-substrate caveat，但 caveat 是事后说明，不如默认配置就用真 substrate。
- **违反**：不违反 R 铁律。但违反 evaluation 公平性精神（"用同 substrate 比较两个 architecture"才是 architectural ablation）。
- **风险**：**中-高**。任何 P5 demo / 内部评估 / 客户尽调引用 smoke 数据会触发 P5 kill criteria 误报；任何后续 commercialization 决策若基于 ΔA3 < 0 误读会偏向"砍 VZ companion"路径。
- **触发条件**：(a) 任何 P5 第三方 demo 引用 SMOKE_REPORT；(b) 任何商业决策 review 引用 ΔA3；(c) 想要真 P5 evidence 时；(d) commercialization §4.5 kill criteria 评估实际触发。
- **推荐修法**：
  1. **立即（不破现有）**：smoke runner 加 `--substrate-mode {synthetic,hf-shared}` 顶层参数 + `--vz-substrate-model-id` 默认 `Qwen/Qwen2.5-1.5B-Instruct`；新加 `SMOKE_PROFILE=cpu_fast | gpu_fair` env，cpu_fast 默认 synthetic 标 "pipeline-only"，gpu_fair 默认 hf-shared 标 "architecture-fair"
  2. SMOKE_REPORT generator ([`_emit_smoke_report.py`](../scripts/companion_bench/_emit_smoke_report.py)) 在 §1 配置表加 `Substrate Mode` 行，gpu_fair 模式才允许进 §3 ΔA3 商业判定，cpu_fast 模式 §3 显示 "Substrate-unfair, no commercial judgement"
  3. 文档 [`docs/external/companion-bench-openrouter-setup.md`](external/companion-bench-openrouter-setup.md) 加 §"Substrate 公平度档级"对照 §"Judge 合格度档级"
  4. 长期：`packages/lifeform-service/` 提供 `--substrate-mode hf-shared` + lazy-load + warm-up cache 让 GPU 加载 < 5 min（当前 ~10-15 min）
- **优先级**：**中**（不阻塞 pipeline 验证 — synthetic 跑通确实证明 wiring 通；但阻塞任何商业 evidence 引用）

---

## 73. Companion Bench SUT subprocess 缺 retry + arc-level fail-isolation

- **路径**：
  - SUT HTTP client: [`packages/companion-bench/src/companion_bench/sut_client.py`](../packages/companion-bench/src/companion_bench/sut_client.py) `OpenAIChatClient.chat`（无 retry，单次 `urllib.request.urlopen`，`request_timeout_s=120` 默认）
  - simulator HTTP client: [`packages/companion-bench/src/companion_bench/user_simulator.py`](../packages/companion-bench/src/companion_bench/user_simulator.py) `OpenAIUtteranceClient.complete`（同上无 retry）
  - judge HTTP client: 同上 (judge 也走 `OpenAIUtteranceClient`)
  - arc runner: [`packages/companion-bench/src/companion_bench/arc_runner.py`](../packages/companion-bench/src/companion_bench/arc_runner.py) `run_arc`（单 turn HTTP 异常 → arc fail）
  - submission orchestrator: [`packages/companion-bench/src/companion_bench/submission.py`](../packages/companion-bench/src/companion_bench/submission.py) `run_submission`（arc 异常未 catch → 整个 SUT 0 bundle）
  - 上游 partial closure: [`#45`](../docs/known-debts.md) `score_reference_systems` subprocess wallclock timeout（已加，本次 25 min 守住）
- **问题**：本次 smoke Phase 2 实测：
  - **第一次跑** (Qwen + VZ): Qwen subprocess 在 user_simulator.py:474 触发 `urllib.error.HTTPError: HTTP Error 400: Bad Request`（DashScope 间歇拒绝 — 可能 token 超限、seed 字段、或 transient server-side），整个 Qwen SUT 0 bundle 写出
  - **第二次跑** (Qwen-only 重跑): 第 1 arc 7 min 完成，第 2 arc 超 25 min subprocess timeout 被 [#45](../docs/known-debts.md) 守门 kill，4 arcs 只完成 1 个，summary.json 没写 → 整个 SUT 数据废
  - 即使有 1 个 bundle 写出，run_submission 没机会调用 `write_submission_summary` → leaderboard 看不到
- **核心含义**：single-vendor reliability 风险（DashScope 当前晚高峰慢、HTTP 400 偶发、token 超限 transient）让整轮跑分白做。严肃 reference run 必须有 retry + arc-level fail-fast。
- **违反**：不违反 R 铁律。但违反 [`docs/external/companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md) reproducibility 精神（同样 manifest + key 应该出同样 result，含 transient 失败容忍）。
- **风险**：**中**。每次 Qwen-only / single-vendor smoke 都有 ~30% 失败率（实测 2 跑 1 全失败 1 部分失败）；正式 reference run（24 scen × 5 SUT × 3 seeds = 360 arcs）只要任一 arc fail 整个 SUT 死，再加 OpenRouter rate limit 风险，nightly small ($200-400) 都跑不出完整 evidence。
- **触发条件**：(a) 任何严肃 reference run；(b) [#34](../docs/known-debts.md) staged executor 重新设计时；(c) DashScope/OpenRouter 间歇性 fail；(d) [`#32`](../docs/known-debts.md) launch 准备真跑 paper-suite-full 时。
- **推荐修法**：
  1. `OpenAIChatClient` + `OpenAIUtteranceClient` 加 `_retry_with_backoff(max_retries=3, backoff_factor=2.0)` helper：429 / 5xx / `URLError` / `TimeoutError` 重试（HTTP 400 不重试 — 是 client 错）
  2. `run_arc` 把单 turn 异常包装为 `ArcExecutionError`，`run_submission` catch ArcExecutionError → log + 跳到下个 arc + 仍写已成功 arcs 的 summary.json（arc-level fail-isolation）
  3. 加 `SubmissionResult.failed_arc_count` + `failed_arc_reasons` 字段，让 leaderboard 显示 "Qwen3-Max: 3/4 arcs OK, 1 failed (HTTP 400)"
  4. `score_reference_systems.py` 在主循环 catch 后写 partial aggregate（即使 0 arc，也写 `{"submission_id": ..., "status": "FAILED", "reason": ...}` 到 aggregate；现在直接 skip aggregate row）
  5. CI workflow [`companion-bench-paper-suite-small.yml`](../.github/workflows/companion-bench-paper-suite-small.yml) 加 `--retry-failed-systems` 把失败 SUT 单独 retry 1 次
- **优先级**：**中**（阻塞 reference run 完整性；smoke 可勉强凑合）

---

## 74. SubmissionManifest.system_prompt + generation_config 在 arc_runner 未注入 SUT messages

- **路径**：
  - schema: [`packages/companion-bench/src/companion_bench/submission.py`](../packages/companion-bench/src/companion_bench/submission.py) `SubmissionManifest.system_prompt: str` + `generation_config: dict` 字段（line 87-99）
  - arc 调度: [`packages/companion-bench/src/companion_bench/arc_runner.py`](../packages/companion-bench/src/companion_bench/arc_runner.py) `run_arc` 调 `sut_client.chat(messages=..., temperature=ArcRunConfig.temperature, max_tokens=ArcRunConfig.max_tokens)` — `messages` 不含 manifest.system_prompt，`temperature` / `max_tokens` 写死 ArcRunConfig 默认
  - 调度入口: [`scripts/companion_bench/run_real_submission.py`](../scripts/companion_bench/run_real_submission.py) line 124-128 构造 `OpenAIChatClient(base_url=..., api_key=..., model=...)` — 不传 manifest.system_prompt 也不读 manifest.generation_config
  - 上游协议: [`docs/external/companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md)（明文 require manifest 的 system_prompt + generation_config 体现在 SUT 上）
- **问题**：`SubmissionManifest` schema 含 `system_prompt: str` + `generation_config: dict`，但运行时**完全没用**：
  - SUT 每 turn 用 vendor 默认 system prompt + ArcRunConfig 默认 generation_config
  - submitter 上传 `system_prompt: "你是友好的 X 顾问"` 与上传 `system_prompt: ""` 跑出**完全相同**的数据
  - manifest 自我描述（"我用了 prompt X + temperature 0.5"）与实际跑分行为脱节
- **核心含义**：reproducibility / 可比性 / 提交方信任三个东西同时坏。任何 prompt-tuning ablation（"我加了 brand voice prompt 后 A2 提升 5 分"）实际不会发生 — 因为 prompt 没注入。
- **违反**：违反 [`docs/external/companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md) reproducibility 协议（spec §3 P3）。
- **风险**：**中**。第一个外部 submitter 想做 system_prompt A/B 时立即发现失效；submission protocol 第一条承诺破坏 → submitter 失去信任。
- **触发条件**：(a) 任何 submitter 提交 manifest 含非空 system_prompt 期望生效；(b) 任何 prompt-engineering ablation 实验；(c) [#32](../docs/known-debts.md) launch 之前必须修；(d) 内部 lifeform-companion SUT 也想用自定义 system_prompt 时（VZ 自己也踩这个坑）。
- **推荐修法**：
  1. `arc_runner.run_arc` 接 `system_prompt: str = ""` + `generation_config: dict = {}` 参数，内部把 system_prompt 注入 `messages = [{"role": "system", "content": system_prompt}, ...] if system_prompt else messages`，把 generation_config 透传 `sut_client.chat(temperature=generation_config.get("temperature", default), max_tokens=generation_config.get("max_tokens", default), ...)`
  2. `submission.run_submission` 透传 `manifest.system_prompt + manifest.generation_config` 到 `run_arc`
  3. 加 `tests/contracts/test_submission_manifest_system_prompt_propagation.py`：fake SUT client 验证 `messages[0]['role']=='system'` 含 manifest.system_prompt 字面量
  4. ArcRunConfig 增加 `inherit_from_manifest: bool = True` 让 default generation_config 来自 manifest 而不是写死
  5. `SUTClient.chat` Protocol surface 已有 `temperature` / `max_tokens` 参数，无需扩 Protocol；只需 caller 真传 manifest 值
- **优先级**：**中**（影响 launch 准入；不影响 pipeline 跑通）

---

## 75. CostTracker.record_perturn_judge / record_arc_judge 在 run_submission 未调用 → judge cost 漏算

- **路径**：
  - cost API: [`packages/companion-bench/src/companion_bench/cost.py`](../packages/companion-bench/src/companion_bench/cost.py) `CostTracker.record_perturn_judge(model, prompt_tokens, completion_tokens)` + `record_arc_judge(...)` 已实现
  - submission orchestrator: [`packages/companion-bench/src/companion_bench/submission.py`](../packages/companion-bench/src/companion_bench/submission.py) `run_submission` **只**调用 `cost_tracker.record_arc_record(arc)` 拿 SUT usage（每 ArcTurn 携带 `sut_prompt_tokens` / `sut_completion_tokens`），**从未**调用 `record_perturn_judge` / `record_arc_judge`
  - judge 实现: [`packages/companion-bench/src/companion_bench/judge_perturn.py`](../packages/companion-bench/src/companion_bench/judge_perturn.py) `LLMPerTurnJudge` + [`judge_arc.py`](../packages/companion-bench/src/companion_bench/judge_arc.py) `LLMArcJudge` — judge `.score()` 内部调 `client_complete(...)` 拿 raw text，**不返回** usage tokens（make_completer 闭包丢失 usage 信息）
  - 价格表: [`cost.py`](../packages/companion-bench/src/companion_bench/cost.py) `_DEFAULT_PRICES`（本次 #71 配套已加 Qwen 系列）
- **问题**：本次 smoke 实测：
  - VZ companion smoke summary `cost.totals.total_usd = 0.0`（lifeform-companion 价格表 = 0，对，但 judge 部分应该有数）
  - dashscope-qwen3-max-smoke v1 summary `cost.totals.total_usd = None`（v1 时 cost.py 没 Qwen 价格 — 已修）
  - 即使 cost.py 加 Qwen 价格表后重跑，**判分用了多少 token、多少钱**仍然 None — 因为 `record_perturn_judge` / `record_arc_judge` 从来没被 call
  - 4 个 arc × ~5 turn × per-turn judge 1 call/turn + arc judge 1 call/arc = ~24 judge calls，全部漏 cost
- **核心含义**：cost evidence 不全 → submitter 看不到 reference judge 真实成本 → [`commercialization-assessment.md`](business/commercialization-assessment.md) §6 P5 单位经济假设（"$5-15k release-tier 跑分"）依赖完整 cost evidence 才能准。
- **违反**：违反 [`docs/external/companion-bench-cost-model-v0.md`](external/companion-bench-cost-model-v0.md) cost transparency 承诺。
- **风险**：**低-中**。不阻塞跑分；但任何 commercialization §6 单位经济回填 / submitter 预算指引 / [#56](../docs/known-debts.md) cost 闭环都要这个数据完整。
- **触发条件**：(a) submitter 看 cost 决定是否值得提交；(b) [#56](../docs/known-debts.md) [`scripts/companion_bench/estimate_quarterly_cost.py`](../scripts/companion_bench/estimate_quarterly_cost.py) 真跑预算分析；(c) [`commercialization-assessment.md`](business/commercialization-assessment.md) §6.5 P5 单位经济回填实测；(d) [#32](../docs/known-debts.md) launch 之前 publish 完整 cost。
- **推荐修法**：
  1. `LLMPerTurnJudge` / `LLMArcJudge` 内部把 `client_complete` 升级为返回 `tuple[str, UsageInfo]`，UsageInfo = `(prompt_tokens, completion_tokens)`；或加 typed `.usage_log: list[UsageEntry]` 字段累积每次 call usage
  2. `submission.run_submission` 在每个 `score_arc_perturn(arc, judge)` 调用后，遍历 judge.usage_log 调 `cost_tracker.record_perturn_judge(model=judge.model, prompt_tokens=..., completion_tokens=...)`；arc judge 同理
  3. 或者更优雅：`make_completer(...)` 闭包 hook CostTracker，在每次 LLM HTTP response 解析后自动 record（避免改 judge 内部）
  4. 加 contract test `tests/contracts/test_companion_bench_judge_cost_recorded.py`：fake judge × N 次 `.score()`，run_submission 后 `result.cost.perturn_calls > 0` AND `result.cost.perturn_usd > 0`
  5. 价格表: [#71](../docs/known-debts.md) 配套已加 Qwen；可顺手补 OpenRouter `<vendor>/<model>` 命名（如 `openrouter/openai/gpt-5-mini`），让档 B (OpenRouter) 路径 cost 自动有数
- **优先级**：**低-中**（与 [#56](../docs/known-debts.md) 同 packet 推进最高效）

---

## 76. 进程级 `default_persona_lora_pool()` 在 Einstein 三 vertical 同进程共存时无法 per-vertical 隔离

- **路径**：
  - synthesizer 端 auto-activate hook: [`packages/lifeform-expression/src/lifeform_expression/llm_synthesizer.py`](../packages/lifeform-expression/src/lifeform_expression/llm_synthesizer.py) `_maybe_activate_persona_lora`（按 `bundle.figure_id` 查 process-wide `default_persona_lora_pool()`）
  - vertical 端 LoRA 注册: [`packages/lifeform-service/src/lifeform_service/verticals.py`](../packages/lifeform-service/src/lifeform_service/verticals.py) `_register_einstein_persona_lora_if_present`（被 `_try_einstein_full` 在 factory 内调用）
  - verification harness 的临时方案: [`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/runtime_conditions.py`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/runtime_conditions.py) `_temporarily_deregister_pool_record`（per-call context manager，对长生命周期 chat session 不适用）
  - pool 实现: [`packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py`](../packages/vz-substrate/src/volvence_zero/substrate/persona_lora_pool.py) `PersonaLoRAPool`（process-level singleton）
- **问题**：本次 F4.3 把 Einstein vertical 拆为 `einstein-raw` / `einstein-bundle` / `einstein-full` 三个 ablation arm，但 LoRA pool 是 **process-wide singleton**：`einstein-full` factory 注册 record 后，pool 状态对同进程所有 vertical 全局可见 → 后续从 `einstein-bundle` 创建的 session 走 `synthesize()` 时，`_maybe_activate_persona_lora` 通过 `bundle.figure_id == "einstein"` 在 pool 中查得到 record → auto-activate，行为与 `einstein-full` 等价，违反 [`PersonaCondition.BUNDLE`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/runtime_conditions.py) 的"persona LoRA 临时摘出"契约。verification harness 用 per-call context manager 解决（call 进入 deregister / call 退出 restore），但 chat UI 的 session 是长生命周期，pool state 必须按 synthesizer 显式 opt-out 而不是按 call 临时摘。
- **当前不影响（debt [#40](../docs/known-debts.md) 守护期下）**：synthetic LoRA backend delta 经 LayerNorm 被吃 → bundle ↔ bundle_lora forward byte-equivalent → pool 共享在 forward 上无副作用；用户在 chat UI 切 vertical 看不出 BUNDLE / BUNDLE_LORA 差异（"看不出来"既掩盖了 #40 也掩盖了 #76）。
- **核心含义**：今天 chat UI 上 demo 的可见差异**全部来自 `einstein-raw` vs `einstein-bundle`**（L4 拒答 / L3 引证 pointer / L1 风格），这是真实的；BUNDLE / BUNDLE_LORA 一对差异在 #40 + #41 closure 之前看不到，在 closure 之后会被 #76 污染。
- **违反**：[`PersonaCondition`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/runtime_conditions.py) ablation contract（runtime 端必须保证 BUNDLE 条件下 LoRA forward 不激活）；ssot-module-boundaries（pool state 是 substrate-side ownership，被 service 层 vertical factory 单向写入是合理的，但被多个 vertical 同时写入而无 per-write隔离不合理）。
- **风险**：当前**低**（debt #40 守护期）→ 一旦 [#40](../docs/known-debts.md) + [#41](../docs/known-debts.md) 闭合**升级到中-高**：真 PEFT-on-Qwen LoRA 注入 pool 后，`einstein-bundle` session 被 `einstein-full` 的 pool registration 静默"污染"，BUNDLE / BUNDLE_LORA 两条 demo 路径无法在 chat UI 上独立展示；任何依赖 byte-level condition isolation 的 SLA / 合同表述（含 [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §6.2 "L1 vs L1+L2 选哪档"客户问询）失效；[`docs/external/companion-bench-rfc-v0.md`](external/companion-bench-rfc-v0.md) reproducibility 协议受影响。
- **触发条件**：(a) [#41](../docs/known-debts.md) 真 PEFT-on-Qwen 跑分完成 + register；(b) BD / 客户法务 demo 需要在同一会话里 A/B 切 BUNDLE vs BUNDLE_LORA 看真差异；(c) 任何依赖 condition byte-level 独立的合同 SLA；(d) [`docs/moving forward/figure-evidence-packet.md`](moving%20forward/figure-evidence-packet.md) Packet 60 voice 盲测的 bundle vs bundle_lora paired t-test 评估需要真隔离样本。
- **推荐修法**：
  1. `LifeformLLMResponseSynthesizer.__init__` 新增 `persona_lora_enabled: bool = True` 字段；`_maybe_activate_persona_lora` 入口 `if not self._persona_lora_enabled: yield; return` 提前 bail；`clone_for_session` / `with_figure_bundle` 透传 flag
  2. 配套 `with_persona_lora_enabled(flag: bool) -> LifeformLLMResponseSynthesizer` clone-style helper（mirror `with_figure_bundle` 模式）
  3. `_try_einstein_bundle` factory 在 `_attach_figure_bundle(...)` 后显式 `bound_synthesizer = bound_synthesizer.with_persona_lora_enabled(False)`；`_try_einstein_full` 保持 default True
  4. 加 contract test [`tests/contracts/test_einstein_vertical_condition_isolation.py`](../tests/contracts/test_einstein_vertical_condition_isolation.py)：3 个 vertical 在同进程 factory(None)，用 fake `LoRAAwareResidualRuntime` mock `activate_lora`；assert `einstein-bundle` 完整 turn 内 `runtime.activate_lora` 调用次数 == 0 而 `einstein-full` >= 1
  5. 同步移除 [`start_browser_chat_qwen.sh`](../start_browser_chat_qwen.sh) + [`.ps1`](../start_browser_chat_qwen.ps1) 头部"Known limitation: bundle ↔ bundle_lora byte-equivalent"段落
- **优先级**：**中**（与 [#41](../docs/known-debts.md) 同 packet 推进最高效；#41 不动则本债性质类似休眠债）

---

## 77. Einstein 训练成果对外只挂在 chat UI，没有离线 3-condition 对照报告（法务尽调 / reviewer 评分材料缺失）

- **路径**：
  - 现有 harness: [`scripts/figure_verify_einstein_persona.sh`](../scripts/figure_verify_einstein_persona.sh) → [`python -m lifeform_domain_figure.verification.persona.cli`](../packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/cli.py) → 输出 `artifacts/figure_verify/<run_id>/{verdict.json, transcript.md, results/<condition>.jsonl, scores.json}`（gate verdict + per-condition transcript，但**未并排**）
  - 新增建议落点: `packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/persona/report_render.py` + `scripts/figure_demo_einstein_report.sh`
  - 与 chat UI 接入的关系: [`packages/lifeform-service/src/lifeform_service/verticals.py`](../packages/lifeform-service/src/lifeform_service/verticals.py) F4.3 已提供 `einstein-{raw,bundle,full}` 三 vertical 现场切换；本债是离线静态产物路径
- **问题**：F4.3 把 demo 主路径设为 chat UI 切 vertical（BOSS 决策 `a_chat_only`），但**离线 evidence** 路径完全缺位：
  - BD / 客户法务首次尽调时只能现场演示，不能离线提交 PDF / markdown
  - [`docs/moving forward/figure-evidence-packet.md`](moving%20forward/figure-evidence-packet.md) Packet 58 / Packet 59 reviewer 标注时只能看 chat UI screenshot
  - [`docs/business/xfund-strategic-thesis.md`](business/xfund-strategic-thesis.md) / [`docs/business/xfund-technical-credibility-brief.md`](business/xfund-technical-credibility-brief.md) 缺一份可附 PDF 的 "raw / bundle / bundle_lora 3 condition × N 题并排" 报告
  - Wave O-P verification harness 已有所有原料（per-condition transcript 与 score 落 `artifacts/figure_verify/`），只缺**渲染器** + Bash 编排把三条 condition 并排
- **核心含义**：这不是 evidence 缺失（Wave K + Wave O-P 已经产出 4-gate verdict + transcript），是 **evidence 展示载体**缺失。今天能离线交付的只有 `verdict.json` 一段 + 三份 `transcript.md`（按 condition 分文件），客户法务看不到"同一题三条件并排"。
- **违反**：无 R 铁律违反；属于"工程交付 vs 商业展示载体"缺口。
- **风险**：**中-低**。不阻塞 chat UI demo（已通），但缺这条会让：(a) [`#58`](../docs/known-debts.md) / [`#59`](../docs/known-debts.md) reviewer 工艺评估时只能看 chat UI screenshot；(b) [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §7.3 P1 GTM 第一击的"我们已经在 N=50 题 reviewed set 上跑出 X" 没有离线打包形态；(c) [`docs/moving forward/figure-evidence-packet.md`](moving%20forward/figure-evidence-packet.md) §1.2 "evidence 不齐"补回的展示路径不完整。
- **触发条件**：(a) 第一次客户法务尽调需要离线 evidence；(b) [`#58`](../docs/known-debts.md) / [`#59`](../docs/known-debts.md) reviewer 标注材料；(c) [`docs/moving forward/figure-evidence-packet.md`](moving%20forward/figure-evidence-packet.md) Packet 60 voice 盲测的 `eval_packet.csv` 渲染（同一渲染器可服务）；(d) 任何"raw vs bundle"的对外 case study。
- **推荐修法**（~0.5 工程天）：
  1. 新 `verification/persona/report_render.py`：读 `artifacts/figure_verify/<run_id>/results/<cond>.jsonl` → group by `question_id` → 每题渲染一段 `### Q{n}: <prompt>\n#### raw\n<answer>\n#### bundle\n<answer> [evidence pointers] [refusal_tag] [voice/cognition score]\n#### bundle_lora\n<answer> [...]` markdown；可选 HTML 输出复用 [`site/assets/charts.js`](../site/assets/charts.js) inline-SVG per-axis bar
  2. `scripts/figure_demo_einstein_report.sh`：复用 `figure_verify_einstein_persona.sh` 跑一次 verification → 调 report_render → 输出 `artifacts/figure_demo_einstein/<run_id>.{md,html}` + 一份 manifest（`bundle_id` / `questions_sha` / `substrate_fingerprint` / `created_at_iso`）
  3. 加 smoke test `packages/lifeform-domain-figure/tests/test_report_render_smoke.py`：deterministic fixture 3 condition × 2 question → render → assert markdown 内含 3 个 condition heading + question text + 至少一个 evidence pointer 标记
  4. 在 [`docs/specs/figure-persona-verification.md`](specs/figure-persona-verification.md) 加一段说明 "离线对照报告" 是 verification CLI 的可选 readout，与 4-gate verdict 同一份原料、不重跑、不影响 verdict 字节稳定性
  5. 与 [#76](../docs/known-debts.md) 同节奏 land 最理想 — #76 让 BUNDLE / BUNDLE_LORA 真分化，#77 让分化看得见
- **优先级**：**中-低**（独立可立刻跑，不依赖 [#41](../docs/known-debts.md) / [#76](../docs/known-debts.md)；与 [`#58`](../docs/known-debts.md) reviewer 工艺 + [`docs/moving forward/figure-evidence-packet.md`](moving%20forward/figure-evidence-packet.md) Packet 58-60 同 packet 推进最高效）

---

## 78. Framework-agnostic `getattr default` pure helpers / cross-tier duck-typed builders 的 typed-path 升级

- **路径**：
  - pure per-source helper（参数 `object | None`）: [`packages/vz-runtime/src/volvence_zero/agent/dialogue_outcome_producers.py`](../packages/vz-runtime/src/volvence_zero/agent/dialogue_outcome_producers.py)（PE / commitment 输入；约 7 处 getattr default） / [`packages/vz-cognition/src/volvence_zero/rupture_state/detection.py`](../packages/vz-cognition/src/volvence_zero/rupture_state/detection.py)（PE / relationship_state 输入；约 9 处 getattr default）
  - cross-tier duck-typed builder: [`packages/vz-cognition/src/volvence_zero/interlocutor/readout.py`](../packages/vz-cognition/src/volvence_zero/interlocutor/readout.py)（`_safe_get` / `_snapshot_value` + 12-axis builder；约 10 处 getattr default） / [`packages/vz-cognition/src/volvence_zero/reflection/engine.py`](../packages/vz-cognition/src/volvence_zero/reflection/engine.py) `_ingest_knowledge_hits` / `_ingest_case_hits`（`domain_knowledge` / `case_memory` 上的 hits[*].hit_id / case_id 访问；约 6 处 getattr default）
  - lifeform-side framework-agnostic helper: [`packages/lifeform-core/src/lifeform_core/followup_manager.py`](../packages/lifeform-core/src/lifeform_core/followup_manager.py) `ingest_commitment_lifecycle` (L162-180) `getattr(entry, "alignment_state", None)` / `advocacy_state` / `followup_policy` 三个 enum 字段 + `_enum_value` helper（注释明确说"Defer an import of the enum to avoid a hard import of the kernel from this lifeform-side module"）
- **问题**：上述模块都通过参数注解 `object | None` / `Any` + `getattr(x, "field", default)` 形式访问 typed snapshot / typed dataclass 字段。模块 docstring 明确说这是**故意 framework-agnostic 设计**（"keep cognition tier free of vz-application type imports"、"Defer enum import to avoid hard kernel import"），目的是让 pure helper / cross-tier builder 不依赖具体 owner-side wheel 类型。这与 [`.cursor/rules/ssot-module-boundaries.mdc`](../.cursor/rules/ssot-module-boundaries.mdc) 的 R8 SSOT "消费者读 typed snapshot 字段" 在字面上不一致。**今天没破系统**：所有上游 owner 仍然按已发布 typed schema 产 snapshot，下游 helper 用 getattr default 拿到的值与 typed access 一致；schema drift 会同时让 typed access AssertionError 和 helper 静默 fallback——后者隐藏了契约破裂。
- **核心含义**：reviewer 在写这些 helper 时刻意权衡了"R8 严格 typed"和"wheel-graph 简洁/独立"，选了后者。但选择是基于"今天没有跨 wheel 类型 dependency injection 机制"的前提；如果未来仓库引入 `vz-contracts` 层（已经存在）+ pure helper 改读 contract 类型而不读 implementation 类型，typed 化的成本就大幅下降。
- **违反**：[`.cursor/rules/ssot-module-boundaries.mdc`](../.cursor/rules/ssot-module-boundaries.mdc) R8 "消费者读 typed snapshot 字段" 字面违反；reviewer 注释自我标注为"故意 framework-agnostic"已经承认这一点；R15 演化角度看，schema drift 在 helper 端是 silent fallback 而非 fail-loud。
- **风险**：**低**（今天）。所有 upstream owner 类型稳定，helper 输出与 typed access 实质一致。中长期升级到**中**：(a) 当 PE / commitment / interlocutor / reflection 任一 typed snapshot schema 演化（加字段 / 改字段 / 删字段），helper 会静默拿到 default 而非 AssertionError；(b) `_extract_*_keys` / `_key_for` / `_entry_description` 第一/四轮已经修过的"多候选属性名猜测"是这条债务最严重的子集，剩下的 getattr default 属于轻量级 silent-fallback；(c) typed access 会让 ruff / mypy 在 schema drift 时直接报错，少一次"运行 evidence run 才发现指标全 0"的隐藏 bug（参考第三轮 trace_collector 的 4 个 PE 字段名错误：silent fallback 5 个月后才被发现）。
- **触发条件**：(a) 任一上游 typed snapshot dataclass 字段名 / 字段类型变更（schema drift）；(b) 引入 mypy / pyright strict 模式覆盖这些模块；(c) `vz-contracts` 层进一步抽出 typed snapshot 子集供 lifeform-* 直接 import（消除"hard kernel import"顾虑）；(d) 任一 cross-tier helper 输出指标在 evidence run 中表现异常（怀疑 silent fallback）；(e) 团队约定收紧"所有 owner-snapshot 消费者必须走 isinstance + typed access"。
- **推荐修法**（分组渐进）：
  1. **第一组（dialogue_outcome_producers / rupture_state/detection）**：参数注解从 `object | None` 改为 `PredictionErrorSnapshot | None` / `CommitmentSnapshot | None` / `RelationshipStateSnapshot | None`；body 内 `getattr(snap, "field", default)` 改为 `if isinstance(snap, ExpectedSnapshot): use snap.field`。两个模块都已经在 `volvence_zero.*` 命名空间，import 不会破 wheel-graph。
  2. **第二组（interlocutor/readout / reflection/engine）**：docstring 明确说 "free of vz-application type imports"。若决定 typed 化，要么 (a) 把相关 snapshot 类型搬到 `vz-contracts`（更彻底的 SSOT 收拢），要么 (b) 在这些 helper 内 `TYPE_CHECKING` import + 运行时仍 duck-typed（typed 只服务 lint，运行时无新依赖）。
  3. **第三组（followup_manager 内部 enum 字段）**：`CommitmentLifecycleEntry.alignment_state / advocacy_state / followup_policy` 都是 typed enum。改为 `if isinstance(entry, CommitmentLifecycleEntry):` 后直接读 `.alignment_state.value`；保留 str fallback 处理 legacy 测试 fixture。lifeform-core 已经 import vz-cognition 多个符号，无新增 wheel 耦合。
  4. 配套 contract test：grep `getattr(x, "...", ` AST 在改动模块内归零（与 `_safe_get` helper 一并删除）；ruff S110 / F821 静态扫描覆盖。
  5. 与 [`#43`](../docs/known-debts.md) Arch Uplift Phase 2 follow-up 节奏一起做最高效（同属 architecture-level R8 收拢）。
- **优先级**：**低**（不阻塞任何 milestone；reviewer 已经显式权衡过；可以等 schema drift 真发生 / mypy strict 启用时再批量收拢）

---

## 79. `apply_metacontroller_evidence` 用 `regime_id == "X"` 字符串硬编码调整 `strategy_priors` 

- **路径**：[`packages/vz-cognition/src/volvence_zero/regime/identity.py:848-877`](../packages/vz-cognition/src/volvence_zero/regime/identity.py) `apply_metacontroller_evidence` 方法
- **问题**：方法按 metacontroller controller_code 的 world / self / shared bias 硬编码 6 个 if/elif 分支，每个分支 reference 写死的 regime_id 字符串名（`"repair_and_deescalation" / "emotional_support" / "problem_solving" / "guided_exploration" / "acquaintance_building" / "casual_social"`）调整 `_strategy_priors`：
  ```python
  if self_bias >= world_bias and self_bias >= shared_bias:
      for regime_id in ("repair_and_deescalation", "emotional_support"):
          self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
  elif world_bias >= self_bias and world_bias >= shared_bias:
      for regime_id in ("problem_solving", "guided_exploration"):
          ...
  ```
  这是 reviewer-defined fallback policy："metacontroller bias 这么分布时增强这几个 regime 的先验"。第二轮已经修了表达层 `prompt_planner` / `response.py` 的 `regime_id == "X"` 硬编码（R14 违反），但 owner 内部这层未触动——因为 owner 拥有自己的 regime list 和 strategy_priors，严格来说不算 R14"regime 当 prompt 标签"。
- **核心含义**：first-principles 角度看这是"硬编码替代涌现学习"的小例。系统目标是让 metacontroller 学会"什么 bias 分布对应什么 regime mix"，但当前是 reviewer 手写映射。这与 [`#43`](../docs/known-debts.md) Arch Uplift / [`#44`](../docs/known-debts.md) SYS-1 长期训练承诺的"系统应该学习而非硬编码"路径在 owner 内部尚未对齐。
- **违反**：[`.cursor/rules/first-principles-not-patches.mdc`](../.cursor/rules/first-principles-not-patches.mdc) "禁止 硬编码行为规则替代系统应该学习的东西"。**不**违反 R14（"regime 不是 prompt 标签"）—— owner 内部 reference 自己的 regime list 是合规的。
- **风险**：**低-中**（今天）。reviewer-defined fallback 在 closed-alpha 期可控；当 5 个 regime 数量稳定 + bias-to-prior 映射稳定时无副作用。中长期升级到**中**：(a) 增加新 regime 时 5 处字符串都要改；(b) 升级 controller_code 维度或语义时硬编码 if/elif 链不能自动适应；(c) [`#44`](../docs/known-debts.md) SYS-1 长期训练 land 后 reviewer-defined fallback 应该让位给 learned policy，但当前无 codified "fallback → learned" 切换路径。
- **触发条件**：(a) 加新 regime（任一）；(b) controller_code 维度从 3 升到 N（任一 metacontroller 升级）；(c) [`#44`](../docs/known-debts.md) SYS-1 出 learned bias-to-prior mapping evidence；(d) 团队约定"owner 内部不再用 regime_id 字符串硬编码"。
- **推荐修法**（按演化阶段）：
  1. **短期（reviewer-defined fallback 收口）**：把 6 个 if/elif 分支拆出到 [`packages/vz-cognition/src/volvence_zero/regime/templates.py`](../packages/vz-cognition/src/volvence_zero/regime/templates.py) 的 `REGIME_TEMPLATES` 旁，新加 `metacontroller_bias_to_prior_adjustments: tuple[(bias_axis: str, regime_id: str, delta: float), ...]` 表；`apply_metacontroller_evidence` 改为查表 + iter。新 regime 加表行即可，无 5 处字符串改动。
  2. **中期（让 owner 暴露 reviewer-defined fallback 是 fallback）**：在 `RegimeSnapshot` 加 `metacontroller_evidence_source: Literal["reviewer_fallback", "learned"]` 字段；ACTIVE → learned 后表层让位。
  3. **长期（learned bias-to-prior policy）**：跟 [`#44`](../docs/known-debts.md) SYS-1 长期训练同 packet，把 `_strategy_priors` 调整逻辑替换为 metacontroller learned head 的 readout。与第二轮 follow-up [`#80`](../docs/known-debts.md) `score_regimes` 用 evaluation 作 input feature 改进同性质——都需要 spec-level 设计。
  4. 配套 contract test：grep `_strategy_priors\["[a-z_]+"\]` 在 prod code 归零（外部 reference regime_id 字符串归零）；template table 作为新的 SSOT。
- **优先级**：**低**（不阻塞任何 milestone；与 [`#43`](../docs/known-debts.md) / [`#44`](../docs/known-debts.md) 同节奏 land 最高效）

---

## 80. `score_regimes` 用 evaluation 作 input feature 而非纯 PE 字段（第二轮 follow-up）

- **路径**：[`packages/vz-cognition/src/volvence_zero/regime/scoring.py`](../packages/vz-cognition/src/volvence_zero/regime/scoring.py) `score_regimes` 函数 + 内部 `_metric(evaluation_snapshot, "...", default=0.4)` 8 个指标查询（task_score / task_pressure / repair_pressure / social_pressure / decision_delegation_pressure / semantic_surface_active / warmth / support_presence / relationship_stability / alert_pressure）
- **问题**：第二轮 R-PE 修复时识别但未修 `score_regimes` 的 evaluation 用法。当时判断"`score_regimes` 输出 candidates 作 regime 选择 readout，不写学习信号"属于合规的 gate-style 用法；但 strict R-PE 角度看，evaluation 进入**当前 turn regime 选择**仍然是 evaluation 影响系统状态的通路——比 `_record_turn_score` / `_update_historical_effectiveness` 那种"写学习信号"轻一档，但 evaluation 字段（warmth / task_pressure 等）通过 `_metric` 查询 + magic-number default 进入 regime score，规则形态与"在 `prompt_planner` fallback 中 `regime_id == "X"`"相似——都是"应该学到的映射用硬编码替代"。
- **核心含义**：R-PE 严格版本是"evaluation 是 readout / gate，绝不进入任何系统状态决策通路"。`score_regimes` 处于灰色地带：它影响当前 turn 的 active_regime（短期状态），但不写持久学习信号。第二轮判断为"borderline OK"，本债追踪"严格化"路径。
- **违反**：[`.cursor/rules/first-principles-not-patches.mdc`](../.cursor/rules/first-principles-not-patches.mdc) R-PE "evaluation 不是学习源头"严格版本——本案不写学习信号但影响选择；[`#43`](../docs/known-debts.md) Arch Uplift "evaluation 单向流"角度看是边缘违反。
- **风险**：**低**（今天）。evaluation 字段（warmth / repair_pressure）是 owner-published typed metric，作为 regime selection input 在 reviewer 看来是合理 feature。中长期：若 evaluation 内部用 LLM judge 或类似 self-preference 通路（[`#48`](../docs/known-debts.md) / [`#71`](../docs/known-debts.md)），evaluation 就部分变成"模型自评 → 自选 regime"——形成 silent feedback loop，违反 R-PE 单向性。
- **触发条件**：(a) evaluation backbone 加入任何 LLM-as-judge readout（[`#48`](../docs/known-debts.md) actuation 后）；(b) regime 选择基线对 evaluation 字段的依赖度量化超过某阈值；(c) 团队约定"score_regimes 必须只读 PE / dual_track / memory / temporal，禁读 evaluation"；(d) [`#44`](../docs/known-debts.md) SYS-1 learned regime selector land 后 evaluation feature 让位给 PE feature。
- **推荐修法**：
  1. 改 `score_regimes` 函数签名：`evaluation_snapshot` 参数保留供 gate-style 检查（如 `structured_alerts` 触发 hard cap）但不进入 score 计算；指标特征改读 PE 派生字段（`prediction_error.task_error` / `relationship_error` / `regime_error` / `action_error`）+ dual_track 字段（`world_track.tension_level` / `self_track.tension_level` / `cross_track_tension`）+ memory snapshot 字段。
  2. 配套 contract test：grep `evaluation_snapshot.*turn_scores` 在 `score_regimes` body 归零；新增 `test_score_regimes_uses_only_pe_dualtrack_inputs.py` AST 守门。
  3. evidence run 验证 regime selection trajectory 与改动前的差异：跑 `lifeform-bench --longitudinal-rounds 5`，对比修改前后 `regime_match` 指标稳定性。
  4. 与 [`#79`](../docs/known-debts.md) / [`#43`](../docs/known-debts.md) Arch Uplift 同 packet 推进。
- **优先级**：**低-中**（依赖 [`#48`](../docs/known-debts.md) 跨家族 judge sweep 量化"evaluation 进入 score 的偏差"后再决策；当前 closed-alpha 期 reviewer-defined evaluation 字段可控）

---

## 81. `runtime_helpers.py` 6 个 `if domain == "X":` hint summary 文本硬编码

- **路径**：[`packages/vz-application/src/volvence_zero/application/runtime_helpers.py:692-738`](../packages/vz-application/src/volvence_zero/application/runtime_helpers.py) `_domain_summary` / `_domain_topic_tags` 两个 helper
- **问题**：两个 helper 用 6 个 `if domain == "X":` 字符串硬编码分支返回写死的 hint summary 文本 + topic_tags（`family_transition / professional_process / career_decision / structured_decision_support / relational_repair / emotional_support_basics`）。第二轮 R8 修复时引入了 `_CITATION_REQUIRED_DOMAINS / _JURISDICTION_SENSITIVE_DOMAINS / _INTERNAL_GUIDE_DOMAINS` 3 个 frozenset 命名常量（替代驱动**逻辑**的 `domain in {...}`），但 692-738 行的 hint summary 文本是 readout 数据（owner 自己定义的描述文本），按 plan 留作 follow-up。
- **核心含义**：当前 6 段写死的英文 hint summary 是 reviewer-curated 文本。它不驱动决策（不是 R8 violation），但形态上是"owner 把自己 schema 关联的描述硬编码"，与 [`#79`](../docs/known-debts.md) regime_id 硬编码同性质——都是"owner 知识用 if/elif 表达而非 typed schema 表达"。
- **违反**：无 R 铁律违反；属于"hint summary 应该是 owner 发布的 typed data 而非 if/elif 函数"风格债。
- **风险**：**低**（今天）。6 段文本稳定，新 domain 时 reviewer 手动加分支即可。中长期升级到**低-中**：(a) 国际化（中文 hint summary）需要全部 hardcode 双语；(b) 新 domain 增加时 reviewer 容易忘改 `_domain_topic_tags` / `_domain_summary` 任一处（缺一致性守门）；(c) 与 `ApplicationBrief.domain_affinity` 已经 typed 化的 SSOT 路径不一致。
- **触发条件**：(a) 引入第二语言（中文 / 双语 hint summary）；(b) 新加 domain（无论从 [`#66`](../docs/known-debts.md) growth-advisor 还是 figure-vertical）；(c) hint summary 文本需要被 evaluation / reflection 消费（不再只是 prompt 渲染）；(d) reviewer 想批量改文体风格（如从 declarative 改 reflective）。
- **推荐修法**：
  1. 把 6 段 hint summary 移到 [`packages/vz-cognition/src/volvence_zero/regime/contracts.py`](../packages/vz-cognition/src/volvence_zero/regime/contracts.py) 的 `ApplicationBrief` 旁，新加 `DomainHintCatalog` dataclass：`summary_per_domain: dict[str, str]` + `topic_tags_per_domain: dict[str, tuple[str, ...]]`；owner 一份发布，application/runtime_helpers 直接读字典。
  2. 多语言时 `DomainHintCatalog` 加 `language: str` 字段，per-language catalog 注册。
  3. contract test：grep `if domain == ` 在 `runtime_helpers.py` 归零；新 domain 加入时 contract test 自动覆盖 summary + topic_tags 一致性。
  4. 与 [`#79`](../docs/known-debts.md) regime fallback table / [`#43`](../docs/known-debts.md) Arch Uplift 同 packet 推进最高效（都属于"owner schema 替代 if/elif"）。
- **优先级**：**低**（不阻塞任何 milestone；与 i18n 路径同节奏 land 最高效）

---

## 82. Companion Bench reference SUT 真跑实证缺位（leaderboard launch 硬前提 / 融资尽调 evidence 阻塞）

- **路径**：[`packages/companion-bench/`](../packages/companion-bench/) v1.0 reference impl + [`scripts/companion_bench/score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) reference SUT orchestrator + [`site/data/submissions/`](../site/data/submissions/) leaderboard 数据面
- **问题**：截至 2026-05-15，Companion Bench v1.0 reference 实现已 land（32 文件完整 package + 24 公开 + 96 held-out scenarios + 6 family × 6 axis），但 **6 大主流 substrate 真实跑分尚未完成实证**：
  - **GPT-5**（OpenAI）/ **Claude Opus 4.7**（Anthropic）/ **Qwen3-Max**（Alibaba）/ **DeepSeek V4** / **Llama 5**（Meta）/ **Gemini**（Google）
  - 其中 [`site/data/submissions/dashscope-qwen3-max-smoke.json`](../site/data/submissions/dashscope-qwen3-max-smoke.json) 已有 Qwen smoke 跑分（受 #71 / #72 影响 evidence 不可外引），其余 5 个主流 substrate **零真实跑分**
- **核心含义**：
  - (a) deck v2.7 Slide 13 (Companion Benchmark) 列出"Reference SUT: GPT-5 / Claude / Qwen / DeepSeek / Llama / Gemini" — 实际**只有 Qwen smoke 跑过，且因 judge robustness 问题（#71 / #72）evidence 不可外引**
  - (b) Patrick / senior VC 视角的关键质问："**你说自己 benchmark 上分数最高 — 但谁来跑你的尺子？大厂为什么要来跑你造的 benchmark？**" — 当前回答只能给出 "Phase A 在跑，预计 X 月公布"，**不能给出实证分数**
  - (c) 这是 Companion Bench 作为"行业可信度资产 / 出题人位置二阶溢价"的**核心 evidence 缺口** — 无 reference SUT 真实跑分 = leaderboard 没有 baseline = "vanity benchmark" 风险
- **违反**：无 R 铁律违反；属于"对外 benchmark 工程层就位 → 实证层未触发"的预算 + 时机层债。
- **风险**：**高**（融资尽调 + 行业可信度视角）。短期：deck v2.7 Q&A 备答"Phase A 跑分中，预计 X 月公布"但被识别为延迟；中期：Patrick DD 团队会问"哪个 academic 已 review 你的方法论？哪个大厂团队已合作？" — 当前两条都缺位；长期：Companion Bench 如未在 6-12 个月内有 ≥ 3 个大厂主流模型真实跑分 + 至少一个 third-party academic backing，会被归为 "early benchmark / vanity scoring" 档位，失去 leaderboard 公信力窗口。
- **触发条件**：(a) Companion Bench v1.0 公开 launch 之前（必须先有 baseline 跑分数据）；(b) Patrick / Xfund DD 阶段询问 reference SUT 实测；(c) 任何第三方 vendor 联系 Volvence 询问 benchmark 准入；(d) academic conference / workshop 投稿（leaderboard paper 必须有 reference 数据）。
- **推荐修法**：
  1. **Phase A.1 (M0-M2)**：完成至少 **3 个主流 reference SUT 真跑**（优先级：Qwen3-Max ✓ 已 smoke / DeepSeek V4 / Claude Opus 4.7）— 真跑 24 公开 scenarios × 3 seeds，跑出 95% CI；预算估计 $1K-3K API 费 + 1 周编排（[`#56`](../docs/known-debts.md) [`estimate_quarterly_cost.py`](../scripts/companion_bench/estimate_quarterly_cost.py) 已就位）
  2. **Phase A.2 (M0-M2 并行)**：解决 [`#71`](../docs/known-debts.md) judge robustness 阻塞（Qwen-内 weak proxy judge σ > 8 / 真 cross-family sweep [`#48`](../docs/known-debts.md)）— 至少跑通 Anthropic + OpenAI 两个 family judge 对比，σ < 3 才能 publish leaderboard
  3. **Phase A.3 (M2-M4)**：补齐剩余 3 个 reference SUT（GPT-5 / Llama 5 / Gemini）真跑 — 预算估计 $2K-5K + 1 月编排（[`#32`](../docs/known-debts.md) [`score_reference_systems.py`](../scripts/companion_bench/score_reference_systems.py) 已就位待批准）
  4. **Phase A.4 (M4-M6)**：争取 **1 个 third-party academic backing**（建议 Yang Liu 学术 network：CMU / Yale / Hanneke 系），把 Companion Bench paper 投 ACL / EMNLP / ICLR workshop — 这是从 "vanity benchmark" 升级到 "industry-standard benchmark" 的最强信号
  5. **Phase A.5 (M6 onwards)**：[`#33`](../docs/known-debts.md) human-eval protocol v0.1 land + [`#35`](../docs/known-debts.md) quarterly rotation 真实启动 — 让 Companion Bench 进入 "live benchmark" 状态而非 "static benchmark"
- **优先级**：**高**（融资尽调 + 行业可信度阻塞条件 / leaderboard launch 硬前提）
- **关联**：
  - 与 [`#29`](../docs/known-debts.md) Companion Bench v1.0 reference impl 直接关联（同一 launch path）
  - 与 [`#32`](../docs/known-debts.md) Companion Bench launch sub-tracks (a) 真 10 reference systems 跑分 + (b) DNS / Pages + (c) heldout deploy key + (d) working group 形成同节奏推进
  - 与 [`#48`](../docs/known-debts.md) 真 cross-family judge sweep 强依赖（先 #48 → 再 #82 出 leaderboard）
  - 与 [`#71`](../docs/known-debts.md)–[`#75`](../docs/known-debts.md) Companion Bench smoke real-run findings 同源 — 都是"跑实证暴露 evidence 缺口"

---

## 83. 6 JV 已签 → in-production 真实 ARR 兑现路径缺位（融资尽调 / 商业 evidence 阻塞条件）

- **路径**：跨多个 lifeform-domain-* package（Mobi 私域 / 高盖伦育儿 / 跨境电商 / UploadLive / Hengyi-Guomao / 30K 海外企业）+ `lifeform-service` alpha-API multi-tenant 路径
- **问题**：截至 2026-05-15，6 JV 全部已签合作协议（含 200K 大客户 / 4500 万粉丝 + 5 万企业客户连接基础），但 **0 个 JV 已产生真实 ARR**。所有 deck / commercialization 文档中的 revenue 数字都是基于"真实 partner audience × 行业基线 conversion rate × 已签 partner 协议的分成结构"的 **conservative projection**，**不是 in-production 实测**。这种缺口在融资尽调阶段会被 senior VC 立即识别（"projected vs realized"是 VC 第一道过滤）。
- **核心含义**：
  - (a) deck Slide 19 (Mobi 单位经济) / Slide 20 (3 年财务全景) / Q&A #3 / Q&A #6 的所有数字均为 projected — 这一点在 deck v2.7 已经诚实化（"全部 projected 但基于真实 audience anchor + conservative conversion 假设"）
  - (b) 但 **DD 阶段 Patrick 团队会要求 in-production evidence**：哪个 JV 已经产生第一笔分账？是哪一天？金额多少？invoice 在哪？
  - (c) 当前**没有一个 JV 能给出这些 in-production evidence**——这是融资 timeline 的硬约束
- **违反**：无 R 铁律违反；属于"商业承诺已写进 deck 但实际 production 滞后"的工程 + 业务双层债。
- **风险**：**高**（融资尽调视角）。短期：deck 自身已 projected 标注 + kill criterion 守门；中期：Patrick 这一类 senior VC 在 DD 时会问"3-6 个月内能否给我 1 个 JV 的真实 in-production ARR"，需要明确 commitment timeline；长期：如果 18 个月后 6 JV 仍 0 in-production，整个 commercialization thesis（Slide 14-20）会被打回"早期 thesis 团队"档位。
- **触发条件**：(a) Patrick / Xfund DD team formal request for in-production ARR evidence（任何 institutional fundraising round 启动时）；(b) Series A 启动前的 milestone gate（deck 承诺"6 JV → 3 in-production + ARR > $1M real"）；(c) JV partner 提出"何时进入正式 launch"问题；(d) 内部 OKR season（每 90 天 progress 备忘 to Xfund）。
- **推荐修法**：
  1. **Phase 1 (M0-M3)**：选 2 个 highest-ROI JV（推荐 Mobi 私域 + 高盖伦育儿）作为 lighthouse，**优先做 in-production 接入**：(a) 完成 Volvence engine → JV partner 系统的真实接口；(b) 跑通 1 周小流量试点（10-100 真实用户）；(c) 第一笔真实分账 evidence；(d) 文档化为 [`docs/business/jv-in-production-evidence.md`](business/jv-in-production-evidence.md) 给 DD 团队
  2. **Phase 2 (M4-M9)**：把 3-4 个 JV 推进到 in-production，**真实 ARR 累计达到 $500K-1M USD** — 这是 deck Slide 22 "12-18 个月里要兑现的 milestone" 的 M7-M12 目标，但需要真实数字落实
  3. **Phase 3 (M10-M18)**：剩余 JV 推进 + 跨境电商 SaaS 启动 + 第 7-10 个 JV 签约 + 启动 Series A（deck 承诺数字 ARR $1M real）
  4. **DD 防御**：在 deck 中所有 projected 数字旁明确标注 "projected based on real audience × conservative baseline"（v2.7 已落地 Slide 19 + Q&A #3）；DD 阶段提供 internal financial model Excel 给 Patrick 团队 verify projection 推算逻辑（v2.7 待决策项 #4）
- **优先级**：**高**（融资尽调阻塞条件 / Series A 启动前必兑现）
- **关联**：与 deck v2.7 Slide 22 (12-18 个月 milestone) 强绑定；与 [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §5 推荐排序相关

---

## 84. Reference Companion Harness baseline 缺位（closed-api 列在测"裸 API 没有 memory"而不是测"模型是不是好的 companion 基底"）

- **路径**：新 wheel [`packages/companion-ref-harness/`](../packages/companion-ref-harness/) Apache 2.0（与 [`packages/companion-bench/`](../packages/companion-bench/) 并列）+ packet [`docs/moving forward/companion-ref-harness-packet.md`](moving%20forward/companion-ref-harness-packet.md) 5 个 sub-packet (H-A/H-B/H-C/H-D/H-E)
- **问题**：截至 2026-05-18，CompanionBench `closed-api` track 有两个结构性洞：(a) [`scripts/companion_bench/reference_systems.yaml`](../scripts/companion_bench/reference_systems.yaml) 把 GPT-5 / Claude Opus 4.6 / Gemini 3 Pro / DeepSeek V3 / Llama-3-70B / Qwen 2.5-72B / Mistral Large 全部按 raw `/v1/chat/completions` 端点配置；(b) [`packages/companion-bench/src/companion_bench/arc_runner.py`](../packages/companion-bench/src/companion_bench/arc_runner.py) 第 217 行 `transcript_messages = _fresh_history()` 在每个 non-S1 session 开始时清空历史 → raw API 跨 session 的 A3（continuity，weight 0.25，RFC §4 最大权重）**结构上**就是 0 分。两个洞叠加 → 当前 `closed-api` 列在测"裸 API 没有 memory"而不是测"模型是不是好的 companion 基底"，VolvenceZero Lifeform 与 GPT-5 同图出现就构成 [`commercialization-assessment.md`](business/commercialization-assessment.md) §10.2 反目标"benchmark 价值归零"的同构红线触发条件。
- **核心含义**：
  - (a) RFC §7.4 + submission protocol §3 已规定三个不能混排的 category（`open-weight` / `closed-api` / `bespoke`），但**操作层**缺一个 vendor-neutral 的标准 agent wrapper 把 raw API 公平地包一层 — 这是 LongMemEval / MemoryBench / LoCoMo 等长程 memory benchmark 的标准做法
  - (b) 第一次公开 leaderboard 一发即被反噬："你给 GPT-5 套一层最起码的 memory wrapper 它能拿多少分？" — 没有 ref-harness baseline 就答不上
  - (c) 没有 ref-harness 切片，"controller 层有多少贡献"无法量化，"我们的 Lifeform 比业界标准 agent infra 强多少"的差异化叙事**无 evidence backing**
- **违反**：无 R 铁律违反；与 R2（稳定基底 + 自适应控制器）+ R8（snapshot SSOT）方向**一致** — 新 wheel 是显式 controller 层；属于"对外 benchmark 方法论层缺一档 fair-comparison baseline"的预算 + 时机层债。
- **风险**：**中-高**（融资尽调 + leaderboard 公信力视角）。短期：H-A SHADOW 代码已 land 不阻塞任何 milestone；中期：launch packet [#32](#32) sub-track 1 真 reference 跑分若在 H-D ACTIVE 前启动，公开 leaderboard 的 closed-api 列就缺 ref-harness 子列 → 媒体 / 学术界第一周内质疑；长期：竞品（OpenAI / Anthropic / Meta / character.ai 类）若先于我们建立 long-session benchmark RFC 并自带 reference harness baseline，CompanionBench convener 窗口关闭。
- **触发条件**：(a) launch packet [#32](#32) sub-track 1 真 reference 跑分批准启动前；(b) deck v2.7+ 任何"GPT-5 vs Lifeform 分数差"图表对外发布前；(c) RFC v0.1 → v0.2 公开评论期收尾（必须含 ref-harness baseline definition）；(d) 任何 third-party academic / 厂商 review 时质问"reference harness 在哪"；(e) 媒体 / 投资人 DD 提出"raw API 跨 session 怎么可能拿分"的合理质疑。
- **推荐修法**（按 packet 5 sub-packet 顺序）：
  1. **H-A ACTIVE evidence**（M0-M1，~$400-600 / 1 周 wallclock）：跑 6 substrate × 24 scenario × 1 seed × 2 slice（raw / summary-only），summary-only A3 提升 ≥ 15 分 ⇒ ACTIVE；落 [`docs/external/companion-ref-harness-h-a-ablation-v0.md`](../docs/external/) ablation report
  2. **H-B embed retrieval**（M1-M2，~2 人周 + ~$500-700）：加 `embed_retrieval.py` + `embed_backend.py`（OSS BGE-base + sqlite-vec / FAISS-CPU；contract test 守门白名单 OSS embedding model），policy 加 retrieval block，跑 F1 (continuity) 家族 callback fabrication 率不上升 + A3 callback_probe 准确率 ≥ 70% ⇒ ACTIVE
  3. **H-C user-model + episodic**（M2-M4，~2.5-3 人周 + ~$1500-2000）：加 `user_model.py` typed schema (`UserFact{key, value, source_turn, confidence}`) + `episodic_memory.py` + extractor confidence ≥ 0.85 守门（避免 harness 主动喂错误记忆）；leave-one-out 5 slice ablation 每组件 ≥ 5 分贡献 ⇒ ACTIVE
  4. **H-D 方法论文档同步**（M3-M4 与 H-C 并行，~1-1.5 人周 / $0）：RFC §7.4 三 category → 5-6 category（`closed-api · raw` / `· ref-harness-summary` / `· ref-harness-full` + `bespoke` + `bespoke · case-study`）；[`docs/external/companion-bench-submission-protocol.md`](external/companion-bench-submission-protocol.md) §3 加 `harness_attestation` 字段；[`scripts/companion_bench/reference_systems.yaml`](../scripts/companion_bench/reference_systems.yaml) 拆 `.raw.yaml` + `.ref_harness_summary.yaml` + `.ref_harness_full.yaml`；[`scripts/companion_bench/run_real_submission.py`](../scripts/companion_bench/run_real_submission.py) 加 `--reference-systems-set raw|ref_harness_summary|ref_harness_full|all` flag；[`archetecture.md`](../archetecture.md) 库清单加 `companion-ref-harness` Apache 2.0
  5. **H-E competitor case study**（M4-M5，~1-1.5 人周 + ~$200-400）：[`docs/external/companion-bench-competitor-case-study-protocol.md`](external/) 落档；优先 Anthropic Claude with Memory + OpenAI Assistants thread（自带 memory 层 + TOS 友好）；Replika Pro / Inflection Pi 走 outreach + 书面 opt-in，拿不到就跳过；**case study 不进 ranked column**（site `bespoke · case-study` 单独 section + 显式 disclaimer）；c.ai / Talkie / Soulmate 无 API → 永不接入 CompanionBench，交给 [#29](#29) / [#30](#30) 人评 arena 路径
- **优先级**：**中-高**（leaderboard 公信力 / 自high 风险预防 / 与 [#82](#82) launch path 同节奏推进）
- **关联**：
  - 与 [#82](#82) Companion Bench reference SUT 真跑实证 **transcript cache 共享**（H-A → H-C 的 6 substrate × 24 scenario × 1 seed = 144 arc 与 #82 phase A.3 同源；合并预算从 $8000 降到 ~$5500）
  - 与 [#48](#48) cross-family judge sweep **复用 judge ensemble**（H-A → H-C ablation 跑分用 #48 已 land 的 judge ensemble）
  - 与 [#32](#32) Companion Bench launch sub-tracks 节奏绑定：H-D ACTIVE **必须早于** sub-track 1 真 reference 跑分启动
  - 与 [#71](#71) / [#72](#72) Companion Bench smoke real-run findings 同源（都是"跑实证暴露 evidence 缺口"，但 #71 / #72 关 judge robustness / synthetic substrate，本债关 fair baseline）
  - 与 [#29](#29) / [#30](#30) 人评 arena 路径**互补不竞争**：本债解决"OpenAI-compat 客观跑分的 fair-baseline 公正性"；[#29](#29) / [#30](#30) 解决"对外可被 google 到的客观分数 + 人评 arena 排名"；c.ai / Replika / Talkie 等无 API 产品永远走 arena 路径
  - 与 [`docs/business/commercialization-assessment.md`](business/commercialization-assessment.md) §10.2 反目标 "公开 companion-bench 的私有 held-out 提示集 → benchmark 价值归零"**同构红线**：本债不直接违反但**直接降低**该红线触发概率（补 ref-harness baseline → leaderboard 第一次公开时少一个 reviewer 质疑点）

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