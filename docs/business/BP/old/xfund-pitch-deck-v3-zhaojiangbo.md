# VolvenceZero — Xfund Pitch Deck v3

> Status: **v3.0 draft (2026-05-17)**
> Purpose: 可直接转 PPT + 口播 script 的冷静版 deck。
> Design principle: **少讲宏大判断，多讲可验证事实；少讲"我们一定赢"，多讲"我们正在用什么证据验证"。**
>
> Recommended meeting format: **37 min presentation + 23 min conversation**.
> If Patrick wants a shorter flow: use Slides 1-12 only, keep Slides 13-17 as backup.

---

## V3 的核心变化

这版不再追求"每页都打爆"。Patrick 这种 senior VC 不需要被情绪推着走，他需要快速判断三件事：

1. **这个 founder/team 是否可信。**
2. **这个 wedge 是否独特且足够尖。**
3. **接下来 6-18 个月能不能用真实 ARR 证明。**

因此 v3 做 5 个降温处理：

- 删除所有内部自评词：如"灵魂级 thesis"、"杀伤力"、"结构性独占"、"唯一"、"OpenAI 永远做不出"。
- 技术 claim 改成可验证表述：从"已经证明下一代范式"改为"我们已有早期工程 evidence，DD 可重跑"。
- 商业预测降级为 scenario：主线不再把 2028 收入当结论，而是讲 6 个已签入口如何转成真实 ARR。
- Founder 叙事保留事实，删除自我解释：让 Patrick 自己判断 first-principle quality。
- Einstein / Companion Bench / O(log n) / Experiment Roadmap 进入 appendix 或 Q&A，不抢私域 wedge 主线。

---

## 一句话版本

**Volvence builds a relationship runtime for AI products that need cross-session memory, adaptive behavior, and auditable governance.**

我们的第一个商业 wedge 是中国私域运营：品牌和 MCN 有大量长期关系池，但现有工具只做触达，不会建立关系。我们已经签了 6 个 JV，接下来 18 个月的核心任务不是讲更多 vision，而是把其中 3 个转成 in-production，跑出 real ARR。

---

## 会议节奏

```text
0-5 min      Founder + team credibility
5-10 min     Why this market now
10-18 min    Product wedge: private traffic relationship runtime
18-25 min    Why our system is different
25-31 min    Traction, milestones, business model
31-34 min    Global expansion + platform option
34-37 min    Risks, ask, close
37-60 min    Conversation / demo / diligence questions
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

We are not trying to build a smarter base model. We use frontier models as substrate. Our layer is the runtime above them: identity, memory, relationship state, adaptation, and governance.

Today I want to show you three things: why this team is credible, why private traffic is our first wedge, and what we need to prove over the next 18 months.

**Design note**

Keep this page extremely clean. No "digital life infrastructure" yet. Let "relationship runtime" and "Thin Prompt. Thick Runtime." land first.

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

**Cut from v2**

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
> A layer above frontier models that maintains:
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

That is the runtime we are building.

**Suggested visual**

Stack diagram:

```text
Applications: private traffic / companion / education / figure
Volvence Relationship Runtime: identity, memory, relationship stage, adaptation, governance
Thin prompt: renders current runtime state into language
Substrate: GPT / Claude / Qwen / DeepSeek / local models
```

---

## Slide 7 — Demo Setup: Mobi Private Traffic JV

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

## Slide 8 — Why This Is Hard

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
> It needs four runtime capabilities:
>
> 1. **Identity stability**: the system does not drift randomly across sessions.
> 2. **Memory discipline**: it remembers what matters and forgets or scopes what should not persist.
> 3. **Adaptation**: behavior changes after feedback and context shifts.
> 4. **Governance**: customers can audit, delete, and constrain behavior.
>
> Prompting can imitate a relationship. A runtime has to maintain one.

**Speaker script (2.5 min)**

This is the technical reason we exist.

A single prompt can imitate warmth, empathy, or a persona. But relationship products need stability across time. They need the system to know who the user is, what has changed, what should be remembered, and what should be constrained.

So the technical problem is not "write a better prompt". Our principle is: prompts should render behavior, not own behavior.

This is the core distinction. In most LLM wrappers, the prompt owns the product logic: persona, memory summary, rules, sales posture, safety constraints. That can work for demos, but it becomes fragile in long relationships.

In Volvence, the durable parts live outside the prompt: relationship state, memory updates, adaptation policy, audit trail, and customer constraints. The prompt should be thin. It should express the current state into language, not secretly contain the whole product.

This is also why we do not frame ourselves as an agent framework. Agent frameworks coordinate tools. Our problem is maintaining a relationship state over time.

**Design note**

This slide replaces the more aggressive "LLM cannot prompt out a real person" claim. The idea remains, but the language becomes defensible: core behavior is state-owned, not prompt-owned.

---

## Slide 9 — Architecture: Body + Brain, In Plain Terms

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

**Speaker script (2.5 min)**

Internally we call the architecture Body + Brain.

Body means stable identity and constraints: what kind of entity this is, what it wants to optimize, what it should avoid, and what relationship posture it should maintain.

Brain means the adaptive layer: memory updates, state transitions, response planning, feedback handling, and auditability.

I want to be careful here. We are not selling anthropomorphic theater. We are using this language because long-term behavior needs something like stable drives and adaptive cognition. The business outcome is simple: more stable, more trustworthy relationship behavior over time.

**Design note**

Keep the diagram simple. Do not show 9 owners or 4 timescale grids in main deck.

---

## Slide 10 — Scientific Base: Learning Under Change

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

Do not say "O(log n) means we only need 1/1000 data" in main deck. If asked, discuss active learning theory carefully as a scientific basis, not a direct business guarantee.

---

## Slide 11 — Evidence We Can Show Today

**On-screen**

> **Technical evidence, in plain English**
>
> We are not asking you to believe a diagram. We can show the system doing four hard things:
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

First, it remembers across sessions. A user says something in session one; after restart, the system can still use it correctly in session two. That matters because a long prompt can look like memory inside one conversation, but it does not prove relationship continuity.

Second, it updates old beliefs. If a user corrects information, the system should stop using the outdated version. This is a bigger deal than retrieval. RAG can retrieve old notes; a relationship system has to revise what it believes.

Third, it keeps users separated. Alice's preferences cannot leak into Bob's behavior. That sounds basic, but it is essential for enterprise and private-traffic deployments.

Fourth, we can test behavior changes safely. New modules can run in shadow before becoming active, so we can compare old behavior and new behavior before customers depend on it.

Fifth, the prompt is thin. Core behavior is not stored in a giant hidden system prompt. The durable logic lives in runtime state: memory, relationship stage, constraints, adaptation, and governance. The prompt mostly renders that state into language.

On the commercial side, we have 6 signed JVs and access to a large audience and enterprise base. But the open gaps are equally important: the JVs have not yet produced real ARR, and conversion uplift is not yet proven.

So the company is not "proven". It is at the point where the next 6-18 months can turn technical proof into business proof.

**Design note**

This page should build trust. It is stronger because it names the gaps.

**Optional live demo flow**

If Patrick wants proof instead of slides, run a 3-minute technical demo:

1. Create a user memory in session 1.
2. Restart or reload the runtime.
3. Ask a related question in session 2.
4. Correct one fact and show the system stops using the old version.
5. Switch user identity and show no preference leakage.

The point is not that the answer is poetic. The point is that the runtime state behaves correctly.

---

## Slide 12 — Traction: 6 JVs, One Validation Plan

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

This replaces the big 2026-2028 financial projection as the main commercial claim.

---

## Slide 13 — Business Model

**On-screen**

> **Business model**
>
> Near term:
>
> - JV revenue share
> - Per-user / per-account service fee
> - Vertical deployment fee where appropriate
>
> Medium term:
>
> - Digital Life-as-a-Service runtime
> - Vertical bundles for private traffic, companion, education, figure/IP
>
> Mobi example:
>
> - Signed JV structure
> - Service fee + profit share
> - Revenue depends on conversion uplift, still under validation

**Speaker script (2 min)**

The near-term model is JV-driven: service fees, revenue share, and vertical deployment fees where appropriate.

For Mobi, the signed structure includes a service fee and profit share. The model can be attractive if conversion uplift is real, but that uplift is not yet proven. That is why we treat the next phase as validation, not scaling.

In the medium term, if multiple verticals repeat, the product becomes Digital Life-as-a-Service: a relationship runtime with vertical bundles.

The business model is promising, but we should earn the right to call it repeatable.

**Optional appendix**

Detailed 3-year financial outlook can remain in appendix as scenario, not main claim.

---

## Slide 14 — Global Expansion: Public-to-Private-to-Platform

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
> Volvence can understand customers through relationship behavior.
>
> **Long-term option:** a relationship-commerce platform, not just a SaaS tool.

**Speaker script (3 min)**

I want to add one strategic point, because it explains why this can expand beyond China.

China private traffic is our first wedge because the behavior already exists at massive scale. Public traffic is pulled into WeChat, groups, and 1-on-1 relationships. Commerce then happens through repeated trust, not just one ad click.

But this pattern is not uniquely Chinese. Globally, public audiences also move into private or semi-private channels: TikTok creators move fans into Discord, WhatsApp, SMS, email lists, communities, memberships, and commerce flows.

The missing layer is the same: once a user leaves the public algorithmic feed and enters a private relationship channel, who understands that user over time? Who remembers their preferences, intent, life context, trust stage, and timing?

TikTok understands attention through content and recommendation behavior. Volvence's opportunity is to understand customers through relationship behavior.

That creates a platform option. The first product is relationship runtime for specific verticals. But if we can repeatedly move users from public audience into private relationship, learn preference and intent over time, and then match supply around that relationship, this becomes more than SaaS. It becomes a relationship-commerce platform.

I would still frame this carefully: this is not the next 12-month proof. The next 12-month proof is production usage and real ARR. But it is the reason the wedge can become much larger if the runtime works.

**Design note**

This page should not sound like "we will build the next TikTok". The comparison is narrower: TikTok owns attention intelligence; Volvence aims to own relationship intelligence after users enter private channels.

---

## Slide 15 — Risks

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

## Slide 16 — The Ask

**On-screen**

> **The Ask**
>
> Raise: **$3M-$5M**
>
> Target structure:
>
> - Late Seed / Pre-Series A
> - Xfund as lead or co-lead
> - 18-month runway
>
> Use of funds:
>
> - Engineering: relationship runtime, deployment reliability, evaluation
> - Product: 3 in-production JV launches
> - GTM: lighthouse customers and partner success
> - Legal / governance: audit, consent, deletion, IP structure
>
> Next financing gate:
>
> - 3 JVs in production
> - ARR > $1M real
> - Repeatable deployment playbook

**Speaker script (2 min)**

We are raising $3M to $5M as a late seed or pre-Series A round, with Xfund as lead or co-lead if there is alignment.

The purpose of this round is not broad expansion. It is validation. We need 18 months to turn signed access into production deployments, real revenue, and repeatable evidence.

The next financing gate is clear: 3 JVs in production, more than $1M real ARR, and a repeatable deployment playbook.

If we hit that, we can raise Series A from a position of evidence. If we do not, we should narrow or rethink the wedge.

**Design note**

Pre-money valuation can be discussed verbally or included only if the founder is comfortable. If shown, keep it as "target range under discussion", not a hard demand.

---

## Slide 17 — Close

**On-screen**

> **The thesis**
>
> AI products that matter over time need relationship infrastructure.
>
> The first wedge is private traffic.
>
> The next 18 months are about proof:
>
> signed access -> production usage -> real ARR.

**Speaker script (45s)**

The thesis is simple.

Many AI products will not be judged by one answer. They will be judged by whether they can maintain a useful, trusted relationship over time.

Volvence is building the runtime for that. Private traffic is our first wedge because the market already has relationship demand and distribution access.

The next 18 months are about proof: signed access to production usage to real ARR.

I would love to spend the rest of the time on your questions.

---

# Optional Appendix / Q&A Slides

These should not be in the 35-minute core flow unless Patrick asks.

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
> - Benchmark design exists
> - Reference SUT runs not complete
> - Judge robustness still under validation

**Speaker script**

We built Companion Benchmark because existing benchmarks mostly measure single-turn preference, task completion, or role-play consistency. Our category needs long-session relationship evaluation.

But I would not overclaim this today. The benchmark is designed, but the reference model runs and judge robustness are still in progress. It is a useful evaluation asset, not yet an industry standard.

---

## Appendix B — Einstein / Figure Bundle

**Use when asked:** "Is this only for private traffic?"

**On-screen**

> **Figure bundle example: Einstein**
>
> Demonstrates:
>
> - Style and voice fidelity
> - Position consistency
> - Source-grounded responses
> - Scope refusal when outside documented material
>
> Strategic use:
>
> - Shows auditability and scope control
> - Relevant to museums, education, publishing, IP holders
> - Not the first commercial wedge

**Speaker script**

Einstein is a useful case study because it shows the runtime's auditability and scope control. A normal persona prompt tries to answer everything. Our figure bundle can refuse when outside documented material.

That matters for institutions like museums or publishers. But I would not make it the center of this fundraising story. The first wedge is private traffic because it has clearer distribution and revenue path.

---

## Appendix C — Active Learning Claim, Carefully Stated

**Use when asked:** "How does Yang's active learning work connect to the product?"

**On-screen**

> **Careful claim**
>
> Active learning theory shows that, under certain assumptions, selected labels can reduce sample complexity versus passive labeling.
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
> If 3-6 JVs enter production and conversion assumptions hold:
>
> - 2026: early ARR from lighthouse deployments
> - 2027: repeatable JV rollout + vertical deployment fees
> - 2028: Digital Life-as-a-Service vertical bundles
>
> All figures should be treated as projections until real ARR is recognized.

**Speaker script**

We do have a detailed internal model, and we can share it during DD.

But I would treat it as scenario planning, not proof. The proof points are production deployments, conversion signal, and recognized ARR.

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

At level one, the system can maintain the same user's relationship state across sessions. That is the minimum requirement.

At level two, it can revise old information after correction. That is the difference between retrieval and learning.

At level three, user states stay isolated. That is required for real deployment.

At level four, new behavior can run in shadow before it affects customers. That gives us a safe way to improve.

At level five, the core behavior works with thin prompts. This is important. We are not saying there is literally no prompt anywhere; every LLM call has some interface instruction. The point is that the prompt does not own the product logic. It renders runtime state.

At level six, we compare old and new behavior with metrics. That is when the system becomes an engineering discipline rather than founder intuition.

The technical moat is not that we have one magical model or one clever prompt. It is the runtime discipline around relationship state.

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

---

## Q5 — Why Xfund?

**Answer**

Xfund is a fit because this is a founder judgment and category formation bet, not a conventional SaaS metrics bet yet.

You also understand the thesis that proprietary vertical data can matter more than competing with base-model scaling. We are extending that thesis into relationship data and relationship state.

But the decision should still be evidence-based: founder quality, scientific depth, signed access, and a clear 18-month proof plan.

---

# PPT Production Notes

## Visual style

- Black or near-black background.
- One main thought per slide.
- Avoid dense tables except Slide 12, Slide 14, and Slide 15.
- Use green only for emphasis, not decoration.
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
- After naming an open gap, pause. Let the honesty register.
- If Patrick interrupts, stop the deck and go into conversation.
- The goal is not to finish all slides. The goal is to create trust.

## Words to avoid

- "唯一"
- "永远"
- "结构性独占"
- "杀伤力"
- "灵魂级"
- "打爆"
- "OpenAI 做不了"
- "已经证明"

## Replacement language

- "我们的当前判断是..."
- "这还没有被完全证明..."
- "接下来 18 个月要验证..."
- "DD 阶段可以现场重跑..."
- "如果这个指标不成立，我们会重新评估..."

