# Business 文档索引

> 本目录存放 VolvenceZero 的商业评估、产品化路线、商业化策略相关文档。
> 与 `docs/prd.md`（产品需求）、`archetecture.md`（架构边界）、`docs/specs/`（能力域 spec）平行：
> - `docs/specs/` 回答"系统应该怎么造"
> - 本目录 回答"系统造出来后应该怎么卖"

## 当前文档

| 文档 | 内容 | 状态 |
|---|---|---|
| [commercialization-assessment.md](./commercialization-assessment.md) | VZ 全面商业化评估：差异化资产、市场结构、6 条路径备选、12-24 个月排序、单位经济、GTM、风险与 kill criteria、OKR 草案、反目标 | v0.1（2026-05-13） |
| [xfund-technical-credibility-brief.md](./xfund-technical-credibility-brief.md) | 对 Xfund (Patrick Chung) 一次首谈用的技术可信度对外裁短版：thesis / 架构第一性承诺 / 8 件 ship 的事 / 与 Delphi / Open Evidence 对标 / anti-claims / 60 秒中英文口头版 | v0.1（2026-05-13） |
| [xfund-strategic-thesis.md](./xfund-strategic-thesis.md) | Xfund 完整战略叙事书面版：业界深度调研 / 与头部互相支撑 / 已 ship 实物 + 算法证明 / IQ + EQ 涌现路径 / 三件事 evidence cascade / anti-claims / 60 秒口头版 | v0.1（2026-05-14） |
| [xfund-pitch-deck-blueprint.md](./xfund-pitch-deck-blueprint.md) | Xfund 投资理念深度研究 + PPT 设计原则反推 + 27 页 slide-by-slide 蓝图 + 30/45 分钟两套演讲编排 + leave-behind packet 清单 | **v0.1 已 deprecated** — 由 v2 替代，研究底稿保留 |
| [xfund-pitch-deck-v11-design-brief.md](./xfund-pitch-deck-v11-design-brief.md) | **V11 designer brief（视觉规范配套文档）**：给 freelance 设计师/in-house designer 的可直接执行 spec，配套 V10 ready-to-present + V11-full master 两个内容版本。一句话定调：*Make this deck look like it could be an article in The Information or a Stripe Press book.* 16 节内容：mood reference 表（Stripe Press / FT / Anthropic PDF / Tufte 应该像 vs YC pitch / McKinsey / SaaS 不应该像）、color tokens 双套（dark `#0B0E13` near-black + warm off-white `#E8E6E1` + sage moss `#6FA08C/#8FC7AC` + amber `#C8985A`；light cream `#F8F6F1` 同结构镜像）、typography 系统（premium GT Sectra + Söhne / 免费 Source Serif 4 + Inter + IBM Plex Mono；11 级 type scale 全部带尺寸/weight/tracking/leading）、12-column grid + 8px vertical rhythm、6 个 page template、**4 个 must-land slide 完整 ASCII wireframe + 视觉 spec**（Slide 4 flywheel 八节点 sage hairline 圆 + 右侧 metric/economic/irreversibility 表 / Slide 7 memory→governance 上半 muted 4-bullet + 下半 7-row 治理维度表 + sage 框 punchline / Slide 12 Phase 1→4 不等宽四列 + Phase 1 sage tint highlight / Slide 15 三栏温度梯度 cool→sage→neutral grey）、Observed footer 签名元素详细 spec（永远 hairline 上方 / italic serif / 缩进 / 永不装框）、4 个 section dividers 90% 留白处理、diagram-level 规则（flywheel 不要 box / Day-30 conversation diff 用 mono code-log / 12-week parenting arc 用同一 mono code-log 视觉押韵）、animation 准入（仅 fade-in 200-400ms 与 cover staggered）与 forbidden 清单（fly-in/dissolve/spin/bounce/3D rotate 等全禁）、light-theme 副本（leave-behind PDF）、Figma 文件结构 + 交付清单（PDF 1920×1080 dark + light，PPTX 仅按需）、**negative spec 17 条 forbidden**（无 logo/无 photo/无 emoji/无 gradient/无 drop shadow/无 blue/无 stock/无 QR/无 3D chart/无 centered body/不允许纯黑纯白等）、**10 条 acceptance criteria**（cover 应产生 3 秒停顿 / Slide 4 不需口头解释就能读懂 / 温度梯度无图标传达 / Bloomberg Businessweek art director 不会羞于交付）、7 天 onboarding sequence。配套 V11-full master 与 V10 18-slide ready-to-present 双版本使用。 | **v1.0（2026-05-17）** |
| [xfund-pitch-deck-v11-full-zhaojiangbo.md](./xfund-pitch-deck-v11-full-zhaojiangbo.md) | **当前 master (super-deck SSOT) — V11-full**：把 V2 到 V10 + April 2026 *Beyond Agents* PDF + 四份底稿（strategic thesis / blueprint / technical brief / commercialization assessment）整合为一份英文为主的 master deck，**未来所有 audience-specific deck 从这里挑选**（不直接 deliver-as-is）。**Hybrid thesis-driven spine**：V10 *memory-is-substrate / governance-is-the-product / parenting-is-the-proof-environment* 在最顶；V9 8-stage trajectory flywheel + V8 *relationship-continuity → economic-lift* 作为 empirical spine；V6 8-step Cognitive AGI 完整 chain（带全部 Proof blocks）作为 Appendix C；V4 / PDF 的 Body+Brain / Soul Migration / Cognitive AI Map / Reverse Validation 作为 Appendix D + Slides 2/3/10/11；V2 Companion Bench Niche Map + Einstein L1–L4 + 25-year first-principle arc 作为 Slides 9/16/17；V3 cooling principles + words-to-avoid + 30/45/60-min 演讲编排集中到 Appendix K。**30 body slides + 11 selectable appendix modules**：A (Yang Liu papers) / B (citations) / C (full 8-step chain) / D (PDF *Beyond Agents* content) / E (engineering reality + 8 things shipped + competitor table) / F (commercialization: 6 paths + Phase A/B/C plan + 10 KRs + anti-goals) / G (18+ Q&A library) / H (3-year financial model with per-vertical breakdown + 2026 cost structure) / I (私域 China structural primer) / J (Patrick Chung + Xfund history + frontier-lab map + 4 macro signals) / K (cool-tone playbook + words-to-avoid + speaking arrangements + leave-behind packet checklist). 含 cross-version provenance table 标注每个 block 的源 V 版本，以及 internal known-inconsistencies 表（高盖伦 11M vs 15M / 8-things-shipped item 8 / Phase A 完成日 / etc.）。V10 不被 V11-full 取代——V11-full 是 **master**，V10 仍是 18-slide ready-to-present Patrick-room default。 | **v11.0-full（2026-05-17）** |
| [xfund-pitch-deck-v10-zhaojiangbo.md](./xfund-pitch-deck-v10-zhaojiangbo.md) | V10 governance-moat + phase-sequencing + uncertainty-boundary 版（18-slide ready-to-present Patrick-room default）。在 V9 trajectory-flywheel 基础上回答两个 senior VC 必问的存在性问题。（1）**substitution risk**：foundation labs 两年后都有 persistent memory 时还剩什么？V10 把 moat 从 memory primitive 移到 *relationship governance* 七维度（what persists / adapts / decays / never transfers / when monetization stops / how rupture is repaired / what consent means longitudinally），并主动承认 foundation memory 会变强、把这件事 reframe 成"它使我们的 substrate 更便宜，不是威胁"。新 Slide 7 + Q1 punchline：*The moat is not remembering more. The moat is deciding what persists, what adapts, what repairs trust, and what becomes economically actionable over years.*（2）**TAM / venture-scale risk**：parenting 够大吗？V10 把 parenting 从 "first wedge" 升级为 *highest-fidelity proof environment*（不是 terminal market），并新增 Slide 12 显式 Phase 1→2→3→4 sequencing（Phase 1 parenting asset proof → Phase 2 Mobi unit-econ proof → Phase 3 enterprise governance defensibility → Phase 4 cross-vertical relationship graph 5-10 年 platform option）。把 "parenting company" 转成 "category company starting in parenting"。（3）**controlled humility**：新 Slide 15 三栏 *proven / strongly suspected / long-term belief*——把不确定性边界自己画清楚。column 2 每条都对应 deck 内 kill criterion 或 falsifier；column 3 明确说"这一列不需要被证明就能让本轮成立"。这是 senior VC 寻找的 founder maturity 信号。Cover 副标题改为 "Memory is substrate. Relationship governance is the product." + "Parenting is the highest-fidelity proof environment."。Q&A 重排：Q1 substitution / Q2 parenting venture-scale 顶到最前。新 Appendix E.4 给 senior 技术 DD partner 提供 substitution-risk 工程级回答（owner-snapshot SSOT / modification-gate primitive / per-jurisdiction scoped deletion with evidence / typed feedback enums / per-vertical bundle compilation 等具体不变量）。 | **v10.0（2026-05-17）** |
| [xfund-pitch-deck-v9-zhaojiangbo.md](./xfund-pitch-deck-v9-zhaojiangbo.md) | V9 trajectory-flywheel 版（新增 8 阶段 flywheel + parenting first wedge + 12-周 parenting arc + ~30% prose 削减）。被 V10 取代——主缺 substitution-risk 主动 reframe + parenting "proof environment vs terminal market" 区分 + uncertainty-boundary controlled-humility 表。完整内容在 V11-full Appendix C + Slides 5/18/19 verbatim 保留。 | v9.0（2026-05-17） |
| [xfund-pitch-deck-v8-zhaojiangbo.md](./xfund-pitch-deck-v8-zhaojiangbo.md) | V8 — "Relationship Continuity Creates Economic Lift" empirical-spine 版。从 manifesto 转向 investment document，trajectory 从哲学词拉回 business language，Mobi 升为 centerpiece，"OpenAI 做不了" 替换为 "frontier labs optimize for generality, we optimize for relationship state"。被 V9 取代——核心缺 trajectory flywheel 与 parenting wedge commitment。 | v8.0（2026-05-17） |
| [xfund-pitch-deck-v7-zhaojiangbo.md](./xfund-pitch-deck-v7-zhaojiangbo.md) | V7 Patrick-Chung-specific 版（founder + thesis-extension + governance）。被 V8 取代。 | v7.0（2026-05-17） |
| [xfund-pitch-deck-v6-zhaojiangbo.md](./xfund-pitch-deck-v6-zhaojiangbo.md) | V6 thesis-driven 版（8-step 论证链 + Proof 块）。Cognitive AGI / token-RL / NL+ETA 等 civilization-level 叙事在 body 主 slide 上。被 V8 取代但完整 8-step 技术 thesis 在 V8/V9 Appendix E 保留。 | v6.0（2026-05-17） |
| [xfund-pitch-deck-v5-zhaojiangbo.md](./xfund-pitch-deck-v5-zhaojiangbo.md) | V5 cool-tone 版（V3 骨架 + V4 三处定向加强：Human-as-vertical-data thesis / Mobi unit econ + kill criterion / Ask 页融资条款）。被 V6 取代。 | v5.0（2026-05-17） |
| [xfund-pitch-deck-v4-zhaojiangbo.md](./xfund-pitch-deck-v4-zhaojiangbo.md) | V4 PDF 配套 markdown（Body+Brain / Soul Migration / Cognitive AI Map / Reverse Validation 等情绪化叙事）。被 V6 取代。 | v4（2026-04） |
| [xfund-pitch-deck-v3-zhaojiangbo.md](./xfund-pitch-deck-v3-zhaojiangbo.md) | V3 cool-tone 第一版骨架（5 个降温原则）。被 V6 取代。 | v3.0（2026-05-17） |
| [xfund-pitch-deck-v2-zhaojiangbo.md](./xfund-pitch-deck-v2-zhaojiangbo.md) | V2 完整版 deck（26 页）：v2.7.2 加强 — **Slide 13 加 Niche Map 对比表**（vs MMLU / Chatbot Arena / MT-Bench / EQ-Bench 3 / RP-Bench / AgentBench — 7 个现有 benchmark 详细对比 + Companion Bench 独占长程关系曲线 niche 明示）+ v2.7.1 — **新增 Slide 8 灵魂 thesis "Human Beings Themselves Are the Vertical Data"** + **Slide 25 加入具体融资条款**（$3-5M / $20-30M pre-money / 7-10% equity）+ Slide 1 Elevator pitch 具体化 + Slide 2 "500 万 fully burned" + Slide 13 Companion Bench reference SUT 诚实标注（known-debts #82 5 phase timeline）+ Slide 23 OpenAI substitution defense + Q&A 全套诚实化 + known-debts #82/#83 入档 + Slide 14 Experiment Roadmap + Slide 13 Einstein Case Study + Slide 11 Hard Evidence + Slide 12 Companion Bench 网站 + 25 年 first principle 弧线 + 私域运营 deep dive + Excel 3 年财务全景 + 12 题 Q&A. 已被 V6 取代但保留为长版参考。 | v2.3（2026-05-15） |

## 阅读顺序建议

**第一次读**：
1. 先读 `commercialization-assessment.md` §1（系统的商业本质）和 §10（不应该做的事）—— 5 分钟内拿到方向感
2. 再读 §4（6 条路径备选）和 §5（推荐排序）—— 知道资源应该怎么分
3. 最后看 §8（风险与 kill criteria）—— 知道什么时候应该停

**做商业决策前**：
1. 先读 §10（反目标）—— 第一道过滤器
2. 读对应路径的章节（§4.x）+ 单位经济（§6）+ GTM（§7）
3. 决策完同步更新 §11 的复盘记录

## 与其他文档的边界

| 这里写 | 不写 |
|---|---|
| 商业模式、定价、客户、GTM | 系统能力、内核架构、契约 |
| 路径概率、风险、kill criteria | R-ID 设计原理、SHADOW/ACTIVE 协议 |
| 单位经济、ARR、毛利率假设 | substrate / owner / snapshot 实现 |
| 反目标、不做什么 | 工程任务、bug、refactor |

工程相关写在 `docs/specs/` 与 `docs/moving forward/`；本目录只写商业判断。

## 复盘节奏

每 90 天复盘一次。复盘输出新建 `commercialization-review-YYYY-MM-DD.md`，不直接 overwrite assessment 主文件；主文件版本号同步更新。
