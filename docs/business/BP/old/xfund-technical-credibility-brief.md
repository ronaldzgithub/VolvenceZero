# VolvenceZero — Technical Credibility Brief for Xfund

> Status: v0.1 (2026-05-13)
> Audience: Patrick Chung (Managing General Partner, Xfund) — 一次首谈所用
> 目标：用 ≤ 10 分钟阅读 + ≤ 60 秒口头复述，让 Patrick 在他自己的投资 thesis 框架内迅速判断"VolvenceZero 是不是值得继续聊"
> 写作纪律：
> 1. 不夸大未跑通的能力（与 [`docs/moving forward/summary.md`](../moving forward/summary.md) 自评保持口径一致）
> 2. 主动给出我们**不**在卖什么（liberal-arts VC 看团队是否"知道分寸"）
> 3. 主动列出与 Xfund 已投项目（Delphi / Open Evidence）的差异化坐标
> 4. 所有"已造出来的事"都对应仓库里 main 分支 + 契约测试守门 + 可现场 demo 的工程现实，不靠 PPT
> 与其他文档关系：本文件是 [`commercialization-assessment.md`](./commercialization-assessment.md) 的对外裁短版；技术细节回链到 [`docs/next_gen_emogpt.md`](../next_gen_emogpt.md) / [`docs/specs/`](../specs/)

---

## 0. TL;DR（半页，可以独立用作 cold outreach）

我们造的不是"更聪明的 LLM"——那是 OpenAI / Anthropic / Google 永远会赢的赛道。

我们造的是 **bounded, continuously adapting digital organism**：一台被工程契约约束的"持续关系型代理运行时"。它具备四个时间尺度的学习循环、连续记忆、双轨（任务/关系）分离、可审计内部状态、可回滚自修改门控、多 vertical 同进程共载、典型企业级合规面（删除/审计/转人工/暂停）。

**为什么 Xfund 应该关心这件事**：

- 与 Patrick 的 AI thesis 一致——startup 不能在 substrate 层赢大厂，要赢只能赢在**架构差异化 + 专有数据 + 行业 hold**。我们的护城河叠在 4 个面（关系记忆、审计 evidence、vertical bundle、benchmark 出题人位置），对手要打需要同时打 4 处。
- 与 Xfund Portfolio 互补——和 **Delphi** 比，他们做 *digital clones*（静态人格快照），我们做 *digital organism*（活的、会随关系演化、有边界、可被治理）；和 **Open Evidence** 比，他们的护城河是 Mayo Clinic 数据，我们的护城河是**架构 + 多 vertical 共载 + 治理面**——同样是"专有数据训练 + 行业 hold"思路的另一切片。
- 工程现实可验证——1100+ 契约测试守门、5 个垂直角色同进程并行（CI 强制）、closed-alpha API 已在跑、行业级长程陪伴 benchmark v1.0 已 Apache 2.0 开源。

后面用 5 个问题展开。每个问题答完，团队都接受被反驳。

---

## Q1. 你们看见了什么别人没看见的？（Thesis）

我们看见的是：**LLM 的天花板是能力 scaling，不是关系连续性 + 治理可证 + 多角色复用**。这三件事不是 prompt 包装能做出来的，必须从架构层做。

### 1.1 拆开来讲

| 大厂的位置 | 我们看见的位置 |
|---|---|
| LLM = 单时间尺度优化（pretrain + RLHF） | 数字生命 = **4 时间尺度嵌套学习**（online-fast / session-medium / background-slow / rare-heavy） |
| 记忆 = RAG / vector store / 简单 KV | 记忆 = **4 stratum 连续谱**（瞬态 / 情景 / 持久 / 派生索引），各有独立 owner |
| 学习 = token 空间端到端梯度 | 学习 = **冻结基底 + 潜空间控制器代码 `z_t`**（不在 token 空间做长期策略学习） |
| Persona = system prompt | Persona = **不可变 bundle**（含 retrieval index / coverage map / style prior / steering / LoRA），可重现可审计 |
| Boundary = guardrail filter | Boundary = **typed `boundary_consent` owner**，4 反销售边界硬约束写进契约测试，不是 prompt |
| Trust = NDA + model card | Trust = **typed audit + handoff queue + scoped delete + deletion evidence**，可被合规观察 |

### 1.2 为什么这件事不是"更好的 prompt"

R8（snapshot-first contracts）是核心：每个有意义的运行时区域有**唯一主所有者**，跨模块只通过不可变快照交换。这不是工程偏好，是**唯一能让"自适应控制器层 + 冻结基底 + 多 vertical 共载"在生产并发下不互相污染**的工程纪律。任何 LLM-on-API 架构都做不到这件事，因为它没有 owner 概念，只有调用栈。

### 1.3 与 Patrick 的 AI thesis 对齐

Patrick 公开表达过的判断：*"startup 不可能在通用 LLM 上竞争；机会在专有数据 + 训练 + 行业 hold"*。

我们的差异化里：
- **专有数据 + 训练**——figure vertical 走"reviewed corpus → bundle 编译 + LoRA bake"路径（已跑通 Einstein bundle，wave A-G land）；growth-advisor 走"reviewed profile + L2 编译"路径（`cheng-laoshi` 已编码）
- **架构差异化**——R8/R10/R12/R15 这套契约式运行时，是**LLM 包装无法事后补**的工程地基

两条护城河叠加，不是单押一条。

---

## Q2. 你们的架构为什么是对的？（3 个第一性承诺 + 业界对齐）

我们不靠堆术语。我们的架构落在 3 个第一性承诺上，每个都有学术锚点 + 业界正在收敛的方向。

### 2.1 承诺 A：关系不是智力副产物 → **双轨学习**

Premack & Woodruff (1978) 的 ToM 经典工作 + Saxe / Wellman 的认知发展研究都说明：**信念、欲望、意图、情感是可分离的潜在状态**。把它们塞进一个 `user_model` bucket，必然会产生"用 Alice 的偏好回答 Bob"这类社会性误归因。

我们的 R7（dual-track）+ R16-R20（multi-party identity / ToM owners）：
- `world_temporal` 与 `self_temporal` 控制器独立；任务 PE 流与关系 PE 流分开
- `belief_about_other` / `intent_about_other` / `feeling_about_other` / `preference_about_other` 4 个 owner，每个 keyed by `interlocutor_id`

**业界对齐**：Anthropic Constitutional AI 系列把"关系性输出"作为独立目标；DeepMind active inference 系正在把 PE 当一级运行时对象。我们做的事和这个 ridge 一致，且更结构化。

### 2.2 承诺 B：学习不是单时间尺度 → **嵌套时间尺度 + 涌现时间抽象**

学术锚点是 2025-2026 两篇直接来源工作：

| 论文 | arXiv | 我们怎么用 |
|---|---|---|
| **Nested Learning (NL)** | arXiv:2512.24695 | 整个系统建模为一组互联的多频率联想记忆；架构与优化器是一个 *neural learning module*，不是分开设计 |
| **Emergent Temporal Abstractions (ETA)** | arXiv:2512.20605 | 在 token 之上维持一层潜在控制器代码 `z_t` + 学习到的切换条件 `β_t`；内部控制不在高维 token 空间做 |

**业界对齐**：
- 冻结基底 + 轻量控制器层 → Anthropic / DeepMind / Thinking Machines 一年内核心安全研究员都在收敛这个方向
- Latent action RL (CoLA / FR-Ponder) → 我们 Internal RL on `z_t` 已 wired，**开放对话验证 in progress**（这是诚实的 frontier 风险，不是已 ship）
- Persona Vectors / SAE 漂移监控（Anthropic）→ 我们 dual-track + regime owner 在 wire

### 2.3 承诺 C：自修改必须可治理 → **snapshot-first + ModificationGate + audit + rollback**

任何在线持续学习系统的核心商业风险是"自修改不可控"。我们的 R10（自修改门控）+ R15（迁移可回滚）：

- 在线层只改控制器层（adapter delta），不改 substrate weights
- rare-heavy artifact 训练走 `ModificationGate`，不能 bypass
- 所有自修改产生 audit log，跨重启可枚举
- bundle / artifact 是 byte-equivalent 可回滚的（contract test 守门）

**业界对齐**：scalable oversight + 弱模型 + 工具（Anthropic / OpenAI alignment）。这是 frontier safety 研究正在收敛的工程实践，我们已经做出来了，不只是 paper.

### 2.4 一句话总结架构定位

> **VolvenceZero 是一个配得上承载 cognitive 数字生命的"运行时容器"——能不能装进强义智能取决于 substrate scaling（这不是我们的赌注），但容器本身（关系连续 / 治理可证 / 多角色共载）已经是 LLM-on-API 架构无法事后补的差异化。**

这句话团队内部反复用来自我校准（出处见 [`docs/moving forward/summary.md`](../moving forward/summary.md) 五）。如实告诉 Patrick：我们不是在卖 AGI，是在卖一台**能被审计的活的关系机器**。

---

## Q3. 你们已经造出来了什么？（8 件 ship 在 main 的事）

不是 PPT，是 main 分支 + 契约测试守门 + 现场可 demo 的事：

| # | 已交付资产 | 工程证据 | 商业含义 |
|---|---|---|---|
| 1 | **Phase 1 arch-uplift** (profile-registry / evaluation-cascade / audit-owner) | 96 new contract tests PASS, 1063+ existing zero regression（[详见](../moving forward/experiment-arch-uplift-phase1-exit-evidence.md)） | 架构纪律不是口号——重构都走 SHADOW → ACTIVE，可回滚 |
| 2 | **5 vertical 同进程并行共载** | `PARALLEL_VERTICAL_PAIRS` CI 强制；`lifeform-domain-{emogpt, coding, character, figure, growth-advisor}` 全部在同一进程跑 | 多角色复用是真的，不是 GPT Store 那种"多实例" |
| 3 | **Companion Bench v1.0** | 24 公开 + 96 私有 held-out scenario，6 family × 6 axis，Apache 2.0 已开源（[`packages/companion-bench`](../../packages/companion-bench)） | 行业可信度资产——长程陪伴评估的"出题人位置"目前没有强对手 |
| 4 | **Closed-alpha API 已在跑** | `lifeform-serve --alpha-enabled`，含 user allowlist + scoped memory deletion + weekly report（[详见](../closed-alpha-api-service.md)） | "我们已经在 serve 真用户" 不是 fundraising 故事 |
| 5 | **Rupture/Repair typed loop** | `vz-cognition.rupture_state` + `/v1/sessions/{id}/dialogue-outcomes` typed enum + durable memory write | 用户说"你 over-directive" → 系统真改下一轮 + 写持久记忆。LLM API 做不到这件事 |
| 6 | **Scoped memory + GDPR/PIPL 删除路径** | `UserIdentity.scope_key` + `DELETE /v1/users/me/memory` + 删除证据 ledger | enterprise / 受监管行业能签字的硬要求 |
| 7 | **OpenAI 兼容 facade（read-only）** | `lifeform-openai-compat` 4 packets land | 任何 OpenAI SDK 客户端零改造接入；同时 read-only 守住"外部 harness 不污染 kernel" |
| 8 | **Figure vertical 全链** | Wave A-G land；Einstein bundle (`figure-bundle:einstein:29eacd226a7cdfd0`) 跑通 corpus → bundle → adopt → activate | 真实人物数字复生的工程闭环已 ship，等的是博物馆/大学客户而不是工程债 |

### 3.1 关于 benchmark 的诚实交代

我们**不**用未跑通的 benchmark 数字做营销。

- EQ-Bench 3 三轨 ablation 的 harness 已 ready（[详见](../external/eqbench3-results-internal.md)），但首次正式跑分**尚未执行**——不会在这份文档里给一个让 Patrick 在尽调时被打脸的数字
- Companion Bench v1.0 是 *我们造的* benchmark；reference SUT 跑分（GPT-5 / Claude Opus 4.7 / DeepSeek / Qwen / Llama × 我们）规划在 Phase A 后期（[详见](../moving forward/companion-bench-public-launch-packet.md)），跑完会同步给 Xfund 内部第一份 raw 数据
- 团队内部对 P5 公开化做的方法论防御（judge robustness sweep / calibration sweep / statistical power / trusted-runner protocol / cost model）已经按 26 条 debt 拆好成 packet——**这套准备本身就是给"benchmark 公信力"上保险**

### 3.2 现场 demo 准备

任何 Tier-1 资产可以在 30 分钟内端到端演示：
- 同一个 closed-alpha 实例切到 figure（Einstein 引证拒答）→ 切到 growth-advisor（4 反销售边界触发）→ 切到 companion（rupture/repair）
- 删除一段对话 → 再问相同问题 → 系统真"忘了" + 删除证据可查
- 跨重启加载 bundle → integrity_hash 校验 → 跑相同 prompt 字节级一致

---

## Q4. 与 Xfund Portfolio 的差异化坐标（必须主动答的对标题）

| 对标方 | 他们的 angle | 我们的 angle | 谁更适合什么场景 |
|---|---|---|---|
| **Delphi**（Xfund 已投，digital clones） | 用 AI 复制人物决策的静态快照 | **活的关系机器**：跨会话适应 + 关系修复 + 多角色共载 + 治理可证 | Delphi 适合"已逝/不在场专家的判断回放"；我们适合"持续陪伴关系 + 可演化角色" |
| **Open Evidence**（Xfund 已投，Mayo 数据） | 专有医疗数据 + 训练做出大厂做不出的 vertical AI | 同思路的另一切片：**专有 corpus（figure）+ 专有 reviewed profile（growth-advisor）+ 行业治理面**；护城河来自架构 + 多 vertical 复用 | Open Evidence 是"医疗垂直 + 数据"；我们是"关系/陪伴/数字复生 多 vertical + 治理 + 数据 + 架构" |
| **OpenAI Memory / Anthropic Projects** | 跨会话用户记忆 | 我们做的是 **4 stratum continuum memory + reflection engine**；用户可被遗忘 + tenant 隔离 + 多 vertical scoped | 大厂的 memory 是 prompt-level；我们是 owner-level + 可审计 |
| **Character.ai / Replika / 国内星野/Talkie** | 角色 IP 库 + UGC 生态 + LTV | 我们不打 C 端流量战；走 B2B2C（博物馆/教育/出版/品牌方）+ 治理价值 | 他们打 DAU；我们打 enterprise + IP holder |
| **Mem0 / Letta / A-Mem** | 通用 RAG memory plugin | 我们不打通用记忆赛道；做的是 *cognition-grade* 多 stratum 记忆 + 与控制器层耦合 | 他们卖 SDK；我们卖 vertical 运行时 |
| **HereAfter / Storyfile / DeepBrain**（数字复生竞品） | 视频/语音外壳 + 静态对话 | **L1 retrieval + L3 引证保真 + L4 拒答**——博物馆/学术客户的法务能签字 | 他们做"演出逼真"；我们做"事实正确性 > 表演逼真" |

**关键句**：我们与 Delphi / Open Evidence **不是竞争关系**——他们各自打的是 *snapshot replica* 与 *vertical data moat* 两条切片；我们打的是 *living organism + governance fabric*，是 Xfund AI thesis 的第三块拼图。

---

## Q5. 我们不在卖什么（Anti-claims）

Liberal-arts VC 看到团队**主动列**这一段会加分。直接抄自团队内部商业评估 [`commercialization-assessment.md`](./commercialization-assessment.md) §1.2，没有为对外做美化：

| 容易被误用的卖点 | 我们为什么主动不卖 |
|---|---|
| "比 GPT/Claude 更聪明" | substrate ceiling 锁死，IQ 路径不是我们的护城河 |
| "AGI 路径" | 架构是 cognitive AGI 的**容器**而非**实现**——把容器当成实现卖会被工程现实打脸 |
| "通用记忆插件" | OpenAI Memory / Mem0 / Letta 已在通用 RAG 记忆赛道，拼通用是输的 |
| "Agent 框架" | LangGraph / AutoGen / Crew 已占住编排层；我们的 contract runtime 是给**自己**用的 |
| "情绪识别 / 共情打分" | 这些是 evaluation readout，不是产品 |
| "AI 心理咨询师" | 牌照 / 责任 / 合规直接踩雷；不在 closed-alpha 安全边界内 |
| "强义 cognitive AGI 12-24 个月内可达" | 团队自评概率 < 5%（[出处](../moving forward/summary.md) §三）；任何超出此口径的承诺都不诚实 |

---

## 附录 A. 60 秒口头版（Patrick 可能让你"用一分钟讲一遍"）

> "我们做的不是更聪明的聊天机器人，是一个**有关系记忆、有自我边界、可被治理观察**的数字生命运行时。
>
> 技术地基有两块：一块是 2025-2026 两篇论文 (Nested Learning + Emergent Temporal Abstractions) 给我们的多时间尺度学习 + 潜空间控制器代码；另一块是 Anthropic / DeepMind / Thinking Machines 这一年都在收敛的方向——冻结基底 + 轻量自适应控制器 + safety-grade audit。
>
> 工程上已经造出来的事：1100+ contract test 守门、5 个垂直角色同进程并行共载、closed-alpha API 已经在 serve 真用户、行业级长程陪伴 benchmark v1.0 已 Apache 2.0 开源。
>
> 商业坐标：与 Delphi 比，我们是'活的'对'静态的'；与 Open Evidence 比，我们是'架构 + 治理 + 多 vertical'对'医疗专有数据'——同一个 Xfund AI thesis 下不同的切片。
>
> 我们不卖 AGI，不卖更聪明的 LLM，不卖通用 memory plugin。我们卖**一台能被合规观察的、活的、跨会话连续的关系机器**——这件事 OpenAI 包装做不到，垂直 SaaS 包装也做不到。"

## 附录 B. 60-second pitch (English)

> "We're not building a smarter chatbot. We're building a **runtime for digital life** — a bounded, continuously adapting agent with relationship memory, self-boundary, and governance you can audit.
>
> Two technical anchors: (1) two 2025-2026 papers (Nested Learning, Emergent Temporal Abstractions) for multi-timescale learning + latent controller codes; (2) the convergence direction Anthropic / DeepMind / Thinking Machines safety researchers have been collapsing onto this past year — frozen substrate + lightweight adaptive controllers + safety-grade audit.
>
> What's already shipped: 1,100+ contract tests gating the architecture, 5 vertical lifeforms co-loaded in one process (CI-enforced), a closed-alpha API serving real users, and an industry-grade long-session companion benchmark v1.0 open-sourced under Apache 2.0.
>
> Positioning vs Xfund's portfolio: against **Delphi**, we're *living organism* vs *static clone*; against **Open Evidence**, we're *architecture + governance + multi-vertical* vs *proprietary medical data* — two slices of the same Xfund AI thesis.
>
> We don't sell AGI. We don't sell a smarter LLM. We don't sell a memory plugin. We sell **the only auditable, living, cross-session relationship runtime** that an LLM-API wrapper cannot retrofit and a vertical SaaS wrapper cannot govern."

## 附录 C. 关键回链

为 Patrick 的尽调团队准备：

| 想看 | 看哪份 |
|---|---|
| 设计原理（R1-R20 + R-PE） | [`docs/next_gen_emogpt.md`](../next_gen_emogpt.md) |
| 全面商业评估（含 6 路径概率 / kill criteria / unit economics / GTM） | [`commercialization-assessment.md`](./commercialization-assessment.md) |
| 团队对自身工程信心的冷静校准 | [`docs/moving forward/summary.md`](../moving forward/summary.md) |
| 已上线的 closed-alpha 服务面 | [`docs/closed-alpha-api-service.md`](../closed-alpha-api-service.md) |
| 架构边界（25 wheel × 3 层） | [`archetecture.md`](../../archetecture.md) |
| Phase 1 arch-uplift 退出 evidence | [`docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md`](../moving forward/experiment-arch-uplift-phase1-exit-evidence.md) |
| Companion Bench RFC v0 | [`docs/external/companion-bench-rfc-v0.md`](../external/companion-bench-rfc-v0.md) |
| 数据契约 + slot 注册表 | [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) |

---

## 变更日志

- **2026-05-13 v0.1**：初稿。基于 Patrick Chung / Xfund 公开 thesis（Future Planet Capital 2024-07 访谈 + Harvard Magazine 2021-02 + xfund.com portfolio）+ 仓库 main 分支真实 ship 资产盘点。下次复盘节奏：与 [`commercialization-assessment.md`](./commercialization-assessment.md) §11 同步（每 90 天）。
