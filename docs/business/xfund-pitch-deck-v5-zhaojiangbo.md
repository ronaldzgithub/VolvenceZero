# VolvenceZero — Xfund Pitch Deck v5

> Status: **v5.0 draft (2026-05-17)**
> Base: v3 骨架（冷静版 / 可验证事实优先）。
> Targeted grafts from v4: (1) "Human as vertical data" thesis 接住 Xfund 自己的 thesis；(2) Mobi unit economics + kill criterion 把商业模式写实；(3) Ask 页加 valuation / ticket / equity 数字。
> Purpose: 可直接转 PPT + 上会的版本。
>
> Recommended meeting format: **~40 min presentation + ~20 min conversation**.
> Short version: Slides 1-14 only, Slides 15-18 备用。

---

## V5 与 V3 / V4 的取舍

V3 的核心判断没有变：Patrick 这种 senior VC 不需要被情绪推着走，他要快速判断 founder / wedge / 18 个月能不能用真实 ARR 证明。所以 V5 仍然遵守 V3 的 5 个降温原则：

- 删除内部自评词（"灵魂级"、"杀伤力"、"结构性独占"、"唯一"、"OpenAI 永远做不出"）。
- 技术 claim 改为"已有早期工程 evidence，DD 可重跑"。
- 商业预测以 scenario 出现，主线讲 6 个签约入口转 ARR。
- Founder 叙事保留事实，删除自我解释。
- Einstein / Companion Bench / O(log n) / Experiment Roadmap 仍留 appendix。

V5 相对 V3 的 3 处定向加强：

1. 新增 Slide 7 "Human as the next vertical data"——这是 V4 最锋利、且直接对位 Xfund "vertical proprietary data > LLM scaling" thesis 的一页，V3 砍掉是过度谨慎。措辞按 V3 风格再降温（不说 "OpenAI structurally cannot own"）。
2. 商业模型页（原 V3 Slide 13）用 V4 Slide 20 的 Mobi unit economics 替换：写出每用户单价、profit share 结构、conversion baseline、projection 和 **kill criterion**。kill criterion 是 senior VC 真正关心的可信度信号。
3. Ask 页（原 V3 Slide 16）加入 V4 Slide 25 的 pre-money / ticket / equity / use of funds 数字。VC 见面不给区间反而显示准备不足，但保留 V3 的口气："target range under discussion"。

V5 相对 V4 砍掉的东西：

- Bitter Lesson 升级 / Cognitive AI Map / Three first principles / Soul Migration / Reverse Validation 等"先讲行业再讲自己"的 slides。Patrick 不需要被教 industry。
- 主线财务三年表（仍在 appendix 作 scenario）。
- 所有"OpenAI structurally cannot own"、"won 真的胜出"、"结构性"类语言。
- 50 min × 26 slides → 40 min × 18 slides。

---

## 一句话版本

**Volvence builds a relationship runtime for AI products that need cross-session memory, adaptive behavior, and auditable governance.**

我们的第一个商业 wedge 是中国私域运营：品牌和 MCN 有大量长期关系池，但现有工具只做触达，不会建立关系。我们已经签了 6 个 JV，接下来 18 个月的核心任务不是讲更多 vision，而是把其中 3 个转成 in-production，跑出 real ARR。

---

## 会议节奏

```text
0-5 min      Founder + team credibility
5-10 min     Why this market now
10-15 min    Product wedge: private traffic relationship runtime
15-20 min    Why human relationships are the next vertical data layer
20-28 min    Why our system is different (demo + architecture)
28-35 min    Traction, unit economics, milestones
35-38 min    Global expansion + platform option
38-40 min    Risks, ask, close
40-60 min    Conversation / demo / diligence questions
```

---

# Main Deck

## Slide 1 — Cover

**On-screen**

> **VOLVENCE**
>
> **A relationship runtime for AI products**
>
> **Thin Prompt. Thick Runtime.**
>
> Cross-session memory · adaptive behavior · auditable governance
>
> Built for verticals where the product is not one answer, but a relationship over time.
>
> Zhao Jiangbo, Founder & CEO  
> Xfund conversation · May 2026

**Speaker script (45s)**

Patrick, thanks for taking the time.

I will keep this simple. Volvence is building a relationship runtime for AI products where the user experience depends on memory, adaptation, and trust over time.

The shortest way to say our technical difference is: thin prompt, thick runtime. We do not put the product logic into a giant prompt. The prompt renders language; the runtime owns identity, memory, relationship state, adaptation, and governance.

We are not trying to build a smarter base model. We use frontier models as substrate. Our layer is the runtime above them.

Today I want to show you four things: why this team is credible, why private traffic is our first wedge, why human relationships are the next vertical data layer, and what we need to prove over the next 18 months.

**Design note**

Keep this page extremely clean. Let "relationship runtime" and "Thin Prompt. Thick Runtime." land first.

---

## Slide 2 — Founder: Why I Am Working On This

**On-screen**

> **Zhao Jiangbo**
>
> - Peking University CS + MBA
> - IBM Japan Research Lab
> - China HP Software Sales GM
> - Alibaba VP Assistant + Public Security Industry Director
> - 3x founder; Haopai reached 300K users with zero paid acquisition
> - Self-funded Volvence R&D with RMB 5M since ChatGPT launch
> - Hands-on engineering commitment: GitHub activity since Nov 2022, DD-verifiable

**Speaker script (2 min)**

I will start with myself, because at this stage you are mostly underwriting founder judgment.

My background has three parts. First, CS and systems training: Peking University CS, IBM Japan Research, and years in enterprise software. Second, sales and market training: HP, Alibaba, Tencent, and large enterprise customers. Third, founder training: I have built products before, including Haopai, which reached 300,000 users with zero paid acquisition.

After ChatGPT launched, I stopped other work and self-funded this direction. The total personal investment has been around RMB 5M, now fully spent on team, engineering, experiments, and JV development.

One thing I want to make concrete: I am not a non-technical CEO outsourcing the product. I have been hands-on in the codebase since late 2022. The GitHub activity is available for DD.

The reason this matters is simple: Volvence sits at the intersection of systems engineering, human relationships, and commercialization. My career prepared me for that intersection, but the next 18 months still have to prove it with real customers.

**Cut from v2/v4**

Do not say "25 years prepared me", "all cognition is independent", "first principle thinker", or high-school physics unless Patrick asks about personal story. These may be true, but they reduce signal in a fundraising room.

---

## Slide 3 — Team: Scientific Depth + Commercial Access

**On-screen**

> **Core team**
>
> **Yang Liu, PhD — Co-founder & Chief Scientist**
>
> - CMU PhD; advised by Avrim Blum and Jaime Carbonell
> - IBM Research, Yale postdoc
> - 40+ papers across active learning, drifting distributions, transfer learning
> - Full-time
>
> **Zhao Jiangbo — Founder & CEO**
>
> - CS + enterprise software + large-account sales + founder
>
> **Wang Cangyu — Co-founder / CSO**
>
> - Media and private traffic commercialization; TikTok China agency experience
>
> **Zhang Chi — Co-founder / CTO**
>
> - Tsinghua CS; long-time engineering partner
>
> **Wu Xiang — Co-founder / CMO**
>
> - 20 years market and enterprise GTM

**Speaker script (2 min)**

The most important team point is Yang Liu.

Yang is a CMU machine learning PhD, advised by Avrim Blum and Jaime Carbonell. Her research is directly relevant to our problem: active learning, drifting distributions, transfer learning, and learning when the environment is not stationary.

This is not decoration for the deck. Our product problem is: how does an AI system keep learning when each user, each relationship, and each business context changes over time? Yang's research background gives us a serious scientific base for that question.

The rest of the team gives us commercialization access and delivery capability: private traffic, enterprise sales, engineering, and market operations. It is a small team, but everyone is full-time.

**Design note**

Avoid "Top-10 active learning" unless independently sourced. CMU + advisors + paper cluster is already strong enough.

---

## Slide 4 — The Market Wedge: Private Traffic

**On-screen**

> **Private traffic is China's relationship-based commerce layer.**
>
> Public traffic acquisition -> WeChat / group / personal account -> repeated 1-on-1 relationship -> conversion and retention
>
> Existing tools optimize reach.
>
> The unsolved layer is relationship quality over time.

**Visual**

Simple funnel:

```text
Public traffic
    -> WeChat add / group join
    -> repeated conversation
    -> trust / preference / timing
    -> purchase / repeat purchase / referral
```

**Speaker script (2 min)**

Our first wedge is private traffic in China.

For an overseas investor, this market can be easy to underestimate. It is not just CRM. It is a relationship-based commerce layer built on WeChat, groups, personal accounts, and repeated 1-on-1 interactions.

Brands and MCNs do not only need to send messages. They need users to feel remembered, understood, and approached at the right time. Existing tools mostly optimize reach: mass messaging, tags, workflows, auto-replies.

The missing layer is relationship quality over time. That is the wedge where our architecture is most directly useful.

**What not to say**

Do not start with "$150B TAM" or "largest untapped vertical". Start with mechanism. If Patrick believes the mechanism, TAM can follow.

---

## Slide 5 — Why Existing Tools Fall Short

**On-screen**

> **The current stack is built for contact, not relationship.**
>
>
> | Current private-traffic tools | What customers actually need      |
> | ----------------------------- | --------------------------------- |
> | Mass messaging                | Remember individual context       |
> | Tags and workflows            | Adapt to relationship stage       |
> | Keyword automation            | Respond to behavior change        |
> | Sales conversion scripts      | Build trust before recommendation |
> | CRM records                   | Cross-session relationship state  |
>
>
> **Our claim:** relationship quality becomes a measurable business variable.

**Speaker script (2.5 min)**

Most tools in this market are useful, but they are built around contact, not relationship.

They help companies reach users, segment users, and automate workflows. But they do not solve the harder question: how should the system behave differently after 20 conversations with the same user?

If a user says, "you are too pushy", the next conversation should reflect that. If a user mentioned a family situation last week, the system should remember it appropriately. If the relationship is not ready for a recommendation, the system should wait.

This is not a better script problem. It requires persistent relationship state, memory, adaptation, and governance. That is why we call Volvence a relationship runtime.

**Design note**

This is where the investor should first understand the business pain. Keep it practical.

---

## Slide 6 — Product: Relationship Runtime

**On-screen**

> **Volvence Relationship Runtime**
>
> ### **Thin Prompt. Thick Runtime.**
>
> Prompt is only the language rendering layer.
>
> Runtime owns the product logic:
>
> - Persistent user and relationship state
> - Cross-session memory and preference evolution
> - Adaptive behavior policies
> - Auditable actions and deletion controls
> - Vertical-specific bundles for different business contexts
>
> Base models are substrate. Prompt renders. Runtime owns.

**Speaker script (2.5 min)**

This is the product definition.

We are not replacing GPT, Claude, Qwen, or DeepSeek. We use them as substrate. The Volvence layer maintains the state that base models do not own: user state, relationship state, adaptation history, and governance.

The simplest way to understand our architecture is: thin prompt, thick runtime.

In a normal LLM wrapper, a lot of product logic is hidden in the prompt: persona, memory summary, behavioral rules, sales posture, safety instructions. That can produce a good demo, but it is fragile. If the prompt changes, the product changes. If the context gets long, behavior drifts. If the customer asks for audit or deletion, you do not have a clean state boundary.

In Volvence, the prompt is only the language rendering layer. The runtime owns the durable logic: identity, memory, relationship stage, constraints, adaptation, and audit trail.

The product question is not "can the model answer this message?" The product question is "does the system know how this relationship has evolved, what it should remember, what it should not say, when to recommend, when to hold back, and how to remain auditable?"

**Suggested visual**

Stack diagram:

```text
Applications: private traffic / companion / education / figure
Volvence Relationship Runtime: identity, memory, relationship stage, adaptation, governance
Thin prompt: renders current runtime state into language
Substrate: GPT / Claude / Qwen / DeepSeek / local models
```

---

## Slide 7 — Why This Is The Next Vertical Data Layer

**On-screen**

> **A natural extension of "vertical proprietary data > LLM scaling".**
>
> ---
>
> **Generation 1 vertical data**
>
> - Mayo Clinic / S&P / legal / academic / professional corpora
> - Existing, recorded, institution-owned
> - Powerful, but eventually licensable, synthesizable, or absorbed by frontier models
>
> **Generation 2 vertical data: the relationship itself**
>
> - Real-time: every interaction creates new data
> - Relational: user × time × context × relationship stage
> - Non-transferable: state lives in the owner's runtime snapshots
> - Long-tail by construction: 100M users = 100M micro-verticals
>
> ---
>
> **Implication**
>
> Stronger base models help our substrate.
> They do not, by themselves, become the persistent relationship owner for any specific brand, MCN, or vertical.

**Speaker script (2.5 min)**

This is the page where I want to connect our wedge to your thesis.

Xfund's framing — that proprietary vertical data matters more than competing with base-model scaling — is the framing we have used internally for two years. The question we asked is: what is the next layer of vertical data after Mayo Clinic, S&P, and the obvious institutional corpora?

Our answer: the relationship itself.

First-generation vertical data is institutional: medical, financial, legal, academic. It is powerful, but it is already-existing, already-recorded, and over time it becomes licensable or synthesizable.

Second-generation vertical data is the relationship between a brand or a service and an individual user, accumulated over time. Every interaction creates new data. The state is not a record, it is a trajectory: user, time, context, relationship stage. It is non-transferable, because it lives in the owner's runtime snapshots. And it is long-tail by construction, because every user-relationship pair is its own micro-vertical.

I want to be careful here. I am not claiming that a strong base model cannot ship memory features. They will, and that helps our substrate. The claim is narrower: a generic assistant is not structured to be the persistent relationship owner for a specific brand, MCN, or service vertical. That is the layer Volvence is building toward.

**Design note**

This is the only slide in the main deck where we make a category-level thesis claim. Keep it one page. Do not stack adjectives. Let the framing land.

---

## Slide 8 — Demo Setup: Mobi Private Traffic JV

**On-screen**

> **Mobi JV: private-traffic digital employee**
>
> Partner:
>
> - 28M-follower MCN
> - Large private traffic pool
> - Human operators cannot maintain high-quality 1-on-1 relationships at scale
>
> Demo watch points:
>
> - Remembers prior context across sessions
> - Keeps user preferences separated
> - Recommends only when relationship stage is appropriate
> - Adjusts after user feedback
>
> Current status:
>
> - JV signed
> - Pilot / launch path in progress
> - Conversion uplift not yet proven

**Speaker script (1 min before video)**

Let me make this concrete with Mobi.

Mobi is a 28M-follower MCN. The business problem is simple: there are too many users and too few human operators to maintain high-quality conversations.

In the demo, please watch four things: cross-session memory, preference separation, recommendation timing, and behavior adjustment after user feedback.

One important caveat before we play it: the JV is signed and the demo is real, but conversion uplift is not yet proven. That is one of the main things this round needs to help us validate.

**Video**

Use a 4-5 min edited demo. Seven minutes is too long unless Patrick is deeply engaged.

**Speaker script after video (90s)**

The point of the demo is not that the AI sounds human in one turn. Many systems can do that.

The point is that the system has state across turns and sessions. It can remember, adapt, and avoid treating every message as a fresh prompt.

This is the behavior we want to measure in production: not only response quality, but relationship quality and conversion over time.

---

## Slide 9 — Why This Is Hard

**On-screen**

> **Relationship AI is not a prompt problem.**
>
> **The core distinction**
>
>
> | LLM wrapper              | Volvence                       |
> | ------------------------ | ------------------------------ |
> | Prompt owns persona      | Runtime owns identity          |
> | Prompt summarizes memory | Runtime owns memory state      |
> | Prompt lists rules       | Runtime owns constraints       |
> | Prompt nudges behavior   | Runtime owns adaptation policy |
> | Prompt is hard to audit  | Runtime produces audit trail   |
>
>
> ### **Prompts should render behavior, not own behavior.**
>
> Four runtime capabilities the system must have:
>
> 1. **Identity stability**: the system does not drift randomly across sessions.
> 2. **Memory discipline**: it remembers what matters and scopes what should not persist.
> 3. **Adaptation**: behavior changes after feedback and context shifts.
> 4. **Governance**: customers can audit, delete, and constrain behavior.
>
> Prompting can imitate a relationship. A runtime has to maintain one.

**Speaker script (2.5 min)**

This is the technical reason we exist.

A single prompt can imitate warmth, empathy, or a persona. But relationship products need stability across time. They need the system to know who the user is, what has changed, what should be remembered, and what should be constrained.

So the technical problem is not "write a better prompt". Our principle is: prompts should render behavior, not own behavior.

In most LLM wrappers, the prompt owns the product logic: persona, memory summary, rules, sales posture, safety constraints. That can work for demos, but it becomes fragile in long relationships.

In Volvence, the durable parts live outside the prompt: relationship state, memory updates, adaptation policy, audit trail, and customer constraints. The prompt should be thin. It expresses the current state into language, not secretly contain the whole product.

This is also why we do not frame ourselves as an agent framework. Agent frameworks coordinate tools. Our problem is maintaining a relationship state over time.

---

## Slide 10 — Architecture: Body + Brain, In Plain Terms

**On-screen**

> **Body + Brain**
>
> **Body**
>
> - Stable identity and constraints
> - Needs / drives / restraint parameters
> - Relationship posture
>
> **Brain**
>
> - Memory and state update
> - Policy and response planning
> - Adaptation from feedback
> - Auditable action layer
>
> The goal is not anthropomorphic theater.  
> The goal is stable behavior over long relationships.

**Speaker script (2 min)**

Internally we call the architecture Body + Brain.

Body means stable identity and constraints: what kind of entity this is, what it wants to optimize, what it should avoid, and what relationship posture it should maintain.

Brain means the adaptive layer: memory updates, state transitions, response planning, feedback handling, and auditability.

I want to be careful here. We are not selling anthropomorphic theater. We are using this language because long-term behavior needs something like stable drives and adaptive cognition. The business outcome is simple: more stable, more trustworthy relationship behavior over time.

**Design note**

Keep the diagram simple. Do not show 9 owners or 4 timescale grids in main deck.

---

## Slide 11 — Scientific Base: Learning Under Change

**On-screen**

> **Why Yang Liu's work matters**
>
> Relationship products face changing distributions:
>
> - Users change preferences
> - Relationship stages evolve
> - Business goals shift
> - Feedback is sparse and noisy
>
> Yang's research base:
>
> - Active learning
> - Drifting distributions
> - Transfer learning
> - Nonstationary learning
>
> This gives Volvence a serious starting point for data-efficient adaptation.

**Speaker script (2.5 min)**

This is where Yang's research background becomes directly relevant.

In relationship products, the environment is not stationary. A user's preference changes. A relationship stage changes. Business context changes. Feedback is sparse, delayed, and noisy.

Yang has worked for years on active learning, drifting distributions, transfer learning, and nonstationary learning. That does not magically solve the product problem, but it gives us a serious scientific base for building systems that adapt with limited data.

I would frame our current evidence as early but real: we have engineering tests for memory, persistence, and multi-timescale adaptation. During DD, we can show the repo, test surfaces, and what has or has not been proven.

**What not to say**

Do not say "O(log n) means we only need 1/1000 data" in main deck. If asked, discuss active learning theory carefully as a scientific basis, not a direct business guarantee. (See Appendix C.)

---

## Slide 12 — Evidence We Can Show Today

**On-screen**

> **Technical evidence, in plain English**
>
> We are not asking you to believe a diagram. We can show the system doing five hard things:
>
>
> | Hard thing                    | Plain-language test                                                                    | Why it matters                                         |
> | ----------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------ |
> | **Remembers across sessions** | User says something in session 1; after restart, system uses it correctly in session 2 | This is not a long prompt pretending to remember       |
> | **Updates old beliefs**       | User corrects prior info; system stops using the outdated version                      | RAG retrieves; relationship systems must revise        |
> | **Keeps users separated**     | Alice's preferences never leak into Bob's behavior                                     | Required for any enterprise/private-traffic deployment |
> | **Can test changes safely**   | New behavior runs in SHADOW before becoming ACTIVE                                     | Customers need reliability, not demo magic             |
> | **Runs with thin prompts**    | Core behavior comes from runtime state, not a giant hidden prompt                      | The product is architecture, not prompt engineering    |
>
>
> Commercial evidence:
>
> - 6 signed JV agreements
> - 45M+ follower connection base
> - 50K+ enterprise customer connection base
>
> Open gaps:
>
> - 0 real ARR from JVs as of this deck
> - Conversion uplift not yet proven

**Speaker script (3 min)**

This is the evidence page, and I want to make the technical part concrete.

We are not asking you to believe a beautiful architecture diagram. We can show the system doing five hard things that normal LLM wrappers struggle with.

First, it remembers across sessions. A user says something in session one; after restart, the system can still use it correctly in session two. A long prompt can look like memory inside one conversation, but it does not prove relationship continuity.

Second, it updates old beliefs. If a user corrects information, the system should stop using the outdated version. This is a bigger deal than retrieval. RAG can retrieve old notes; a relationship system has to revise what it believes.

Third, it keeps users separated. Alice's preferences cannot leak into Bob's behavior. That sounds basic, but it is essential for enterprise and private-traffic deployments.

Fourth, we can test behavior changes safely. New modules can run in shadow before becoming active, so we can compare old behavior and new behavior before customers depend on it.

Fifth, the prompt is thin. Core behavior is not stored in a giant hidden system prompt. The durable logic lives in runtime state. The prompt mostly renders that state into language.

On the commercial side, we have 6 signed JVs and access to a large audience and enterprise base. But the open gaps are equally important: the JVs have not yet produced real ARR, and conversion uplift is not yet proven.

So the company is not "proven". It is at the point where the next 6-18 months can turn technical proof into business proof.

**Optional live demo flow**

If Patrick wants proof instead of slides, run a 3-minute technical demo:

1. Create a user memory in session 1.
2. Restart or reload the runtime.
3. Ask a related question in session 2.
4. Correct one fact and show the system stops using the old version.
5. Switch user identity and show no preference leakage.

The point is not that the answer is poetic. The point is that the runtime state behaves correctly.

---

## Slide 13 — Traction: 6 JVs, One Validation Plan

**On-screen**

> **From signed access to real ARR**
>
> Current:
>
> - 6 signed JVs
> - 3 private-traffic related
> - 3 other verticals: companion, parenting, cross-border commerce / enterprise
>
> 18-month validation plan:
>
>
> | Timeline | Milestone                      | Success criteria                                        |
> | -------- | ------------------------------ | ------------------------------------------------------- |
> | M0-M3    | First lighthouse in production | Real users, real usage, measurable retention/conversion |
> | M4-M9    | 3 JVs in production            | Repeatable deployment process, early revenue            |
> | M10-M18  | ARR > $1M real                 | Not projected; recognized revenue                       |
>
>
> If we do not hit these, we should not raise a Series A.

**Speaker script (3 min)**

The main commercial question is whether signed access becomes real revenue.

We have 6 JVs. That is not the same as ARR. I want to be explicit about that. Our 18-month plan is to turn at least 3 of them into in-production deployments and reach more than $1M real ARR.

The first three months should produce the first lighthouse: real users, real usage, and measurable retention or conversion. By month nine, we should have three JVs in production. By month eighteen, we should have real ARR, not projected ARR.

If we do not hit those milestones, we should not be raising a Series A on this story.

**Design note**

This replaces the big 2026-2028 financial projection as the main commercial claim. Detailed scenario lives in Appendix D.

---

## Slide 14 — Business Model: Mobi Unit Economics + Kill Criterion

**On-screen**

> **Mobi private-traffic JV: unit economics**
>
> Signed JV revenue structure (per converted user, per year):
>
>
> | Item                            | Unit                                       | 2026 target scale | Volvence revenue contribution |
> | ------------------------------- | ------------------------------------------ | ----------------- | ----------------------------- |
> | Service fee / token procurement | RMB 30 / user / year                       | ~187K orders      | ~RMB 2.8M                     |
> | JV profit share                 | RMB 100 / user / year distributable profit | same              | ~RMB 2.8M                     |
> | **Mobi JV 2026 subtotal**       |                                            |                   | **~RMB 5.6M (~$800K)**        |
>
>
> ---
>
> **Conversion assumption (projected, not proven)**
>
> - Industry baseline (SCRM, public reporting): ~0.3%
> - Volvence projected with relationship runtime: 0.6-1.0%
> - Pilot data not yet through full 3-month observation window
> - **Kill criterion: if 3-month pilot conversion < 0.5%, this vertical is deprioritized**
>
> ---
>
> **Repricing thesis**
>
>
> | Weimob / Youzan         | Volvence                            |
> | ----------------------- | ----------------------------------- |
> | Reach tools             | Relationship engineering            |
> | Broadcast / auto-reply  | Remembered and understood           |
> | One-time conversion     | Cross-session LTV                   |
> | ~RMB 1K / month / brand | RMB 5K-50K / month / brand (target) |
>

**Speaker script (3 min)**

This is the page where I want to be specific about money, and equally specific about what would make us walk away.

The Mobi JV has two revenue layers per converted user per year: a service fee of about RMB 30 that covers our platform and token procurement, and a profit share of about RMB 100 on the distributable margin. At a 2026 target of around 187,000 converted orders, that contributes roughly RMB 5.6M, about $800K, to Volvence from this single JV.

I have to be explicit about the conversion assumption. The industry baseline for SCRM-style private traffic is around 0.3%. Our projected uplift with the relationship runtime is 0.6% to 1.0%. That projection is anchored on the demo behavior — recommendation timing, preference separation, repair-after-rupture — but the 3-month pilot observation window is not yet complete. So this is a projection, not a result.

The kill criterion is the part I want you to remember. If the 3-month pilot lands below 0.5%, we deprioritize this vertical. We do not double down to defend the thesis. That discipline applies to every JV in the portfolio.

The underlying commercial thesis is not "we are a cheaper Weimob". It is repricing. Weimob and Youzan sell reach, so their ARPU is around RMB 1K per brand per month. We are selling long-term relationship optimization, which gives us a path to RMB 5K-50K per brand per month. That is where the multiple comes from, if the conversion thesis holds.

**Design note**

`projected` and `kill criterion` must be visually unmissable on the slide. They are the credibility signals for this page.

---

## Slide 15 — Global Expansion: Public-to-Private-to-Platform

**On-screen**

> **The platform opportunity: public audience -> private relationship -> supply orchestration**
>
> China private traffic is the first wedge because the behavior already exists at scale.
>
> The same pattern exists globally:
>
>
> | Layer                         | China wedge                           | Global analogues                                       |
> | ----------------------------- | ------------------------------------- | ------------------------------------------------------ |
> | **Public domain**             | Douyin / Xiaohongshu / public traffic | TikTok / Instagram / YouTube / podcasts / newsletters  |
> | **Private relationship**      | WeChat / groups / 1-on-1              | WhatsApp / Discord / SMS / email / creator communities |
> | **Relationship intelligence** | Long-term user state and trust curve  | Preferences, intent, timing, life context              |
> | **Supply orchestration**      | JV partners and vertical supply       | Brands, creators, service providers, local commerce    |
>
>
> TikTok understands attention through content behavior.  
> Volvence's opportunity is to understand customers through relationship behavior.
>
> **Long-term option:** a relationship-commerce platform, not just a SaaS tool.

**Speaker script (3 min)**

I want to add one strategic point, because it explains why this can expand beyond China.

China private traffic is our first wedge because the behavior already exists at massive scale. Public traffic is pulled into WeChat, groups, and 1-on-1 relationships. Commerce then happens through repeated trust, not one ad click.

But this pattern is not uniquely Chinese. Globally, public audiences also move into private or semi-private channels: TikTok creators move fans into Discord, WhatsApp, SMS, email lists, communities, memberships, and commerce flows.

The missing layer is the same: once a user leaves the public algorithmic feed and enters a private relationship channel, who understands that user over time? Who remembers their preferences, intent, life context, trust stage, and timing?

TikTok understands attention through content behavior. Volvence's opportunity is to understand customers through relationship behavior.

That creates a platform option. I would still frame this carefully: this is not the next 12-month proof. The next 12-month proof is production usage and real ARR. But it is the reason the wedge can become much larger if the runtime works.

**Design note**

This page should not sound like "we will build the next TikTok". The comparison is narrower: TikTok owns attention intelligence; Volvence aims to own relationship intelligence after users enter private channels.

---

## Slide 16 — Risks

**On-screen**

> **Main risks and how we test them**
>
>
> | Risk                                                   | What would prove / disprove it                                                          |
> | ------------------------------------------------------ | --------------------------------------------------------------------------------------- |
> | JV access does not convert to usage                    | First lighthouse fails to retain users or produce conversion signal                     |
> | Relationship quality does not improve business metrics | A/B or cohort data shows no uplift over SCRM baseline                                   |
> | Base models add stronger memory                        | We prove value in vertical state, governance, and business workflow, not generic memory |
> | Regulatory / safety constraints tighten                | Audit, deletion, consent, and scoped memory become mandatory product features           |
> | Team execution bandwidth                               | 3 in-production JVs by M9, otherwise narrow focus                                       |
>
>
> The next round is about converting these risks into evidence.

**Speaker script (2.5 min)**

These are the risks I would focus on if I were in your seat.

First, signed JVs may not convert to real usage. Second, relationship quality may not produce business uplift. Third, base models may add stronger memory. Fourth, regulation may tighten. Fifth, a small team may be stretched too thin.

Our answer is not to hand-wave these risks. It is to test them quickly. If we cannot get lighthouse usage, retention, conversion signal, and three in-production deployments, then the story is not ready for Series A.

On substitution risk: if base models get better memory, that helps the substrate. Our value has to be vertical relationship state, governance, and workflow integration. That is the proof we need to produce.

**Design note**

Do not claim OpenAI "cannot" do persistent relationship. Say our product must prove value beyond generic memory.

---

## Slide 17 — The Ask

**On-screen**

> **The Ask — Late Seed / Pre-Series A**
>
>
> | Dimension                      | Target                                           |
> | ------------------------------ | ------------------------------------------------ |
> | Round size                     | **$3M-$5M USD**                                  |
> | Pre-money valuation (range)    | **$20M-$30M USD** *(under discussion)*           |
> | Xfund target ticket            | **$1.5M-$2.5M**, lead or co-lead                 |
> | Equity to Xfund                | **~7-10%**                                       |
> | Runway                         | **18 months**                                    |
>
>
> Use of funds:
>
> - **Engineering 40%** — relationship runtime, deployment reliability, evaluation
> - **Compute / data 25%** — substrate, evidence pipeline, benchmark runs
> - **GTM 20%** — 3 in-production JV launches, lighthouse customers, partner success
> - **Operations / legal / IP 15%** — audit, consent, deletion, IP structure
>
> Next financing gate:
>
> - **3 JVs in production**
> - **ARR > $1M real**
> - **Repeatable deployment playbook**

**Speaker script (2 min)**

We are raising $3M to $5M as a late seed or pre-Series A round, with Xfund as lead or co-lead if there is alignment.

On valuation, the target range we are working with is $20M to $30M pre-money. I want to flag that as a range under discussion, not a take-it-or-leave-it number. We would expect Xfund's ticket in the $1.5M to $2.5M band, which corresponds to roughly 7% to 10% equity at that range.

The purpose of this round is not broad expansion. It is validation. We need 18 months to turn signed access into production deployments, real revenue, and repeatable evidence.

The next financing gate is clear: 3 JVs in production, more than $1M real ARR, and a repeatable deployment playbook.

If we hit that, we can raise Series A from a position of evidence. If we do not, we should narrow or rethink the wedge.

---

## Slide 18 — Close

**On-screen**

> **The thesis**
>
> AI products that matter over time need relationship infrastructure.
>
> Human relationships are the next vertical proprietary data layer.
>
> The first wedge is private traffic.
>
> The next 18 months are about proof:
>
> signed access -> production usage -> real ARR.

**Speaker script (45s)**

The thesis is simple.

Many AI products will not be judged by one answer. They will be judged by whether they can maintain a useful, trusted relationship over time.

Volvence is building the runtime for that. Relationship data — the second-generation vertical data — is what accumulates inside that runtime. Private traffic is our first wedge because the market already has relationship demand and distribution access.

The next 18 months are about proof: signed access to production usage to real ARR.

I would love to spend the rest of the time on your questions.

---

# Optional Appendix / Q&A Slides

These should not be in the 40-minute core flow unless Patrick asks.

---

## Appendix A — Companion Benchmark

**Use when asked:** "How do you evaluate relationship quality?"

**On-screen**

> **Companion Benchmark**
>
> Goal: evaluate long-session relationship behavior, not single-turn preference.
>
> Current design:
>
> - 24 public scenarios
> - 96 held-out private scenarios
> - 30-turn arcs
> - Axes: relationship continuity, adaptation, boundary maintenance, temporal fidelity, theory of mind, regime stability
>
> Current status:
>
> - Benchmark design exists, open-sourced under Apache 2.0
> - Reference SUT runs not complete
> - Judge robustness still under validation

**Speaker script**

We built Companion Benchmark because existing benchmarks mostly measure single-turn preference, task completion, or role-play consistency. Our category needs long-session relationship evaluation.

But I would not overclaim this today. The benchmark is designed, but the reference model runs and judge robustness are still in progress. It is a useful evaluation asset, not yet an industry standard.

---

## Appendix B — Einstein / Figure Bundle

**Use when asked:** "Is this only for private traffic?" or "How do I know this is not just a wrapper?"

**On-screen**

> **Figure bundle example: Einstein**
>
> Shipped artifact: `figure-bundle:einstein:29eacd226a7cdfd0` (immutable, reproducible, auditable)
>
> Four fidelity layers:
>
>
> | Layer       | What it means                      | Architecture evidence       |
> | ----------- | ---------------------------------- | --------------------------- |
> | L1 Voice    | Sounds like him                    | Body / style prior          |
> | L2 Position | Agrees on topics he wrote about    | Soul Migration              |
> | L3 Citation | Substantive claims trace to source | Memory + grounded decoder   |
> | L4 Refusal  | Refuses outside documented scope   | ScopeRefuser + coverage map |
>
>
> Strategic use:
>
> - Demonstrates auditability and scope control
> - Relevant to museums, education, publishing, IP holders
> - Not the first commercial wedge

**Speaker script**

Einstein is a useful case study because it shows the runtime's auditability and scope control. A normal persona prompt tries to answer everything. Our figure bundle can refuse when outside documented material.

The L4 refusal layer is what matters most. Museums, universities, and publishers do not buy theatrical realism — they buy governed fidelity.

That said, I would not make this the center of this fundraising story. The first wedge is private traffic because it has clearer distribution and revenue path.

---

## Appendix C — Active Learning Claim, Carefully Stated

**Use when asked:** "How does Yang's active learning work connect to the product?"

**On-screen**

> **Careful claim**
>
> Active learning theory shows that, under certain assumptions, selected labels can reduce sample complexity versus passive labeling.
>
> Reference: Hanneke & Yang, *Minimax Analysis of Active Learning*, JMLR 2015.
>
> Product relevance:
>
> - Relationship data is sparse and noisy
> - Feedback is expensive
> - Vertical partners do not have internet-scale labeled datasets
>
> What we still need:
>
> - Internal task-level ablations
> - Passive vs active learning curves
> - Production feedback loops

**Speaker script**

I would state this carefully.

Yang's work gives us a serious foundation for data-efficient learning under changing conditions. In theory, active learning can substantially reduce label complexity under specific assumptions.

But I would not claim that theory directly guarantees a 1/1000 data advantage in every vertical. What we need to show in DD and production are task-level ablations: passive versus active learning curves, label efficiency, and product metrics.

---

## Appendix D — 3-Year Financial Scenario

**Use when asked:** "What could this become if the JV model works?"

**On-screen**

> **Scenario, not proof**
>
>
> | Metric               | 2026    | 2027     | 2028     |
> | -------------------- | ------- | -------- | -------- |
> | Revenue RMB          | 35M     | 165M     | 408M     |
> | USD equivalent       | ~$5M    | ~$23.6M  | ~$58.3M  |
> | Net margin           | 31%     | 46%      | 54%      |
> | Project gross margin | 55%     | 65%      | 75%      |
>
>
> Drivers:
>
> - 3-6 JVs entering production
> - Mobi-style unit economics repeating across verticals
> - Substrate cost amortization
> - Fixed-cost dilution
>
> All figures should be treated as projections until real ARR is recognized.

**Speaker script**

We have a detailed internal model, and we can share it during DD.

The number to anchor on is 2026 revenue of around RMB 35M, or roughly $5M. The 2027 and 2028 figures depend on whether the unit economics from Mobi repeat across other verticals.

The point of this table is not "we will do $58M in 2028". The point is that the projection has anchors: signed partner audience size, industry baseline conversion, and the contracted revenue share structures. I would still treat all of it as scenario planning, not proof. The proof points are production deployments, conversion signal, and recognized ARR.

---

## Appendix E — Technical Proof Ladder

**Use when asked:** "How do I know this is not just a wrapper?"

**On-screen**

> **Technical proof ladder: Thin Prompt, Thick Runtime**
>
>
> | Level | What we prove                                      | Investor translation                                           |
> | ----- | -------------------------------------------------- | -------------------------------------------------------------- |
> | L1    | Same user, same relationship state across sessions | It can remember beyond one chat                                |
> | L2    | Corrections update future behavior                 | It can learn, not just retrieve                                |
> | L3    | Different users stay isolated                      | It is deployable in real customer environments                 |
> | L4    | New behavior can run SHADOW before ACTIVE          | It can improve without breaking production                     |
> | L5    | Core behavior works with thin prompts              | Prompt renders; runtime owns                                   |
> | L6    | Metrics compare old vs new behavior                | It can become an engineering discipline, not founder intuition |
>
>
> Identity, memory, relationship stage, constraints, adaptation, and audit live in the runtime.  
> The prompt is only the final language interface.

**Speaker script**

If someone asks whether this is just a wrapper, I would answer with this ladder.

L1: same user's relationship state across sessions. Minimum requirement.

L2: revise old information after correction. The difference between retrieval and learning.

L3: user states stay isolated. Required for real deployment.

L4: new behavior runs in shadow before it affects customers. A safe way to improve.

L5: core behavior works with thin prompts. The point is not that there is literally no prompt anywhere; every LLM call has some interface instruction. The point is that the prompt does not own the product logic — it renders runtime state.

L6: compare old and new behavior with metrics. This is when the system becomes an engineering discipline rather than founder intuition.

The technical moat is not one magical model or one clever prompt. It is the runtime discipline around relationship state.

---

# Q&A Prep

## Q1 — You have 6 JVs but no ARR. Why should I believe this is traction?

**Answer**

You should not treat it as proven revenue. You should treat it as signed distribution access.

That is still meaningful, because this category needs real users and vertical contexts. But the next proof is production usage and recognized ARR. Our 18-month plan is built around that conversion.

---

## Q2 — Why not just use GPT with memory?

**Answer**

Generic memory helps, but relationship products need more than memory. They need vertical state, relationship stage, adaptation policy, auditability, deletion, consent, and business workflow integration.

If GPT memory becomes stronger, our substrate improves. Our product still has to prove value in the relationship runtime above the substrate.

---

## Q3 — What is your unfair advantage?

**Answer**

Three things together:

First, scientific depth in learning under changing distributions through Yang Liu.

Second, commercial access to private traffic and JV partners.

Third, a runtime architecture built around relationship state rather than single-turn response quality.

None of these alone is enough. The combination is what we are testing.

---

## Q4 — What would make you change direction?

**Answer**

If the first lighthouse does not produce real usage, or if relationship quality does not move retention/conversion versus baseline, we should narrow the vertical or rethink the product.

If by M9 we cannot get 3 JVs into production, the issue is execution or market readiness. We should not keep expanding the story without evidence.

The Mobi-specific kill criterion is concrete: 3-month pilot below 0.5% conversion → deprioritize that vertical.

---

## Q5 — What is the real upside, not the SaaS upside?

**Answer**

Two layers.

Near-term: repricing. Reach tools sell at around RMB 1K per brand per month. A relationship runtime that improves long-term LTV is worth RMB 5K-50K per brand per month if conversion holds.

Long-term: the relationship state itself becomes a data asset. Every brand that runs on Volvence accumulates a non-transferable trajectory of user × time × context × relationship stage. That is the second-generation vertical data we described on Slide 7. The platform option, not the SaaS option, comes from being the runtime that owns that accumulation across verticals.

---

## Q6 — Why Xfund?

**Answer**

Xfund is a fit because this is a founder judgment and category formation bet, not a conventional SaaS metrics bet yet.

You also articulate the thesis that proprietary vertical data can matter more than competing with base-model scaling. We are extending that thesis into relationship data — into the relationship itself as the next vertical data layer.

The decision should still be evidence-based: founder quality, scientific depth, signed access, and a clear 18-month proof plan.

---

# PPT Production Notes

## Visual style

- Black or near-black background.
- One main thought per slide.
- Dense tables only on Slides 12, 14, 15, 16, 17.
- Use green only for emphasis, not decoration.
- On Slide 14, `projected` and `kill criterion` must be visually unmissable.
- On Slide 17, the ask box uses thick border + large numbers, but the word "range" stays visible.
- No hype words on-screen.

## Demo handling

- Keep Mobi demo to 4-5 minutes.
- Add English subtitles if conversation is Chinese.
- Highlight only 4 moments:
  - prior context remembered
  - preference separation
  - recommendation timing
  - adaptation after feedback

## Speaker behavior

- Do not read long notes.
- After naming an open gap or kill criterion, pause. Let the honesty register.
- If Patrick interrupts, stop the deck and go into conversation.
- The goal is not to finish all slides. The goal is to create trust.

## Words to avoid

- "唯一"
- "永远"
- "结构性独占"
- "杀伤力"
- "灵魂级"
- "打爆"
- "OpenAI 做不了" / "structurally cannot own"
- "已经证明"

## Replacement language

- "我们的当前判断是..."
- "这还没有被完全证明..."
- "接下来 18 个月要验证..."
- "DD 阶段可以现场重跑..."
- "如果这个指标不成立，我们会重新评估..."
- "kill criterion 是..."
