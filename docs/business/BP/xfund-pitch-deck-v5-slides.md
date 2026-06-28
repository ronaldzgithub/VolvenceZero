---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: '#0a0a0a'
color: '#f0f0f0'
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', 'PingFang SC', 'Microsoft YaHei', 'Segoe UI', sans-serif;
    padding: 56px 84px;
    font-size: 24px;
    line-height: 1.5;
  }
  h1 {
    color: #ffffff;
    font-size: 52px;
    font-weight: 700;
    margin: 0 0 0.35em 0;
    letter-spacing: -0.01em;
  }
  h2 {
    color: #ffffff;
    font-size: 34px;
    font-weight: 600;
    margin: 0 0 0.6em 0;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #222;
  }
  h3 {
    color: #00d68f;
    font-size: 26px;
    font-weight: 600;
    margin: 1em 0 0.5em 0;
  }
  h4 {
    color: #d0d0d0;
    font-size: 22px;
    font-weight: 600;
    margin: 0.8em 0 0.3em 0;
  }
  p, li { color: #e8e8e8; }
  strong { color: #00d68f; font-weight: 600; }
  em { color: #f0f0f0; font-style: italic; }
  blockquote {
    border-left: 3px solid #00d68f;
    padding: 0.2em 0 0.2em 1em;
    color: #c8c8c8;
    margin: 0.6em 0;
    font-size: 0.95em;
  }
  ul, ol { margin: 0.4em 0 0.4em 1.2em; }
  li { margin: 0.2em 0; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.72em;
    margin: 0.5em 0;
  }
  th, td {
    border: 1px solid #2a2a2a;
    padding: 7px 11px;
    text-align: left;
    vertical-align: top;
  }
  th {
    background-color: #161616;
    color: #00d68f;
    font-weight: 600;
  }
  tr:nth-child(even) td { background-color: #101010; }
  code {
    background-color: #161616;
    color: #00d68f;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
  }
  pre {
    background-color: #0e0e0e;
    border-left: 3px solid #00d68f;
    padding: 14px 18px;
    font-size: 0.68em;
    line-height: 1.45;
    overflow: visible;
  }
  pre code { background: transparent; color: #d0d0d0; padding: 0; }
  hr {
    border: 0;
    border-top: 1px solid #222;
    margin: 0.8em 0;
  }
  section::after {
    color: #555;
    font-size: 14px;
  }
  /* Cover slide */
  section.cover {
    justify-content: center;
    align-items: center;
    text-align: center;
  }
  section.cover h1 {
    font-size: 96px;
    letter-spacing: 0.05em;
    margin-bottom: 0.5em;
  }
  section.cover h2 {
    border: none;
    font-size: 32px;
    color: #d0d0d0;
    font-weight: 400;
  }
  section.cover h3 {
    font-size: 38px;
    color: #00d68f;
    margin: 1em 0;
  }
  /* Dense slides with multiple tables */
  section.dense { font-size: 20px; padding: 40px 70px; }
  section.dense table { font-size: 0.62em; }
  section.dense h2 { font-size: 28px; margin-bottom: 0.4em; }
  section.dense h3 { font-size: 20px; margin: 0.6em 0 0.3em 0; }
  /* Kill criterion emphasis */
  .kill {
    color: #ff6b6b;
    font-weight: 700;
    border: 2px solid #ff6b6b;
    padding: 8px 14px;
    border-radius: 4px;
    display: inline-block;
    margin: 0.4em 0;
  }
  .projected {
    color: #ffb84d;
    font-weight: 600;
  }
  /* Ask slide */
  section.ask td:nth-child(2) {
    color: #00d68f;
    font-weight: 600;
    font-size: 1.05em;
  }
  /* Section dividers */
  section.divider {
    justify-content: center;
    align-items: center;
    text-align: center;
    background-color: #050505;
  }
  section.divider h1 {
    font-size: 64px;
    color: #00d68f;
  }
  /* Footer brand */
  .brand {
    position: absolute;
    bottom: 24px;
    left: 84px;
    font-size: 13px;
    color: #555;
    letter-spacing: 0.1em;
  }
---

<!-- _class: cover -->
<!-- _paginate: false -->

# VOLVENCE

## A relationship runtime for AI products

### Thin Prompt. Thick Runtime.

Cross-session memory · adaptive behavior · auditable governance

Built for verticals where the product is not one answer,
but a relationship over time.

<br>

**Zhao Jiangbo**, Founder & CEO
Xfund conversation · May 2026

<!--
Patrick, thanks for taking the time.

I will keep this simple. Volvence is building a relationship runtime for AI products where the user experience depends on memory, adaptation, and trust over time.

The shortest way to say our technical difference is: thin prompt, thick runtime. We do not put the product logic into a giant prompt. The prompt renders language; the runtime owns identity, memory, relationship state, adaptation, and governance.

We are not trying to build a smarter base model. We use frontier models as substrate. Our layer is the runtime above them.

Today I want to show you four things: why this team is credible, why private traffic is our first wedge, why human relationships are the next vertical data layer, and what we need to prove over the next 18 months.
-->

---

## Founder: Why I Am Working On This

### Zhao Jiangbo

- Peking University CS + MBA
- IBM Japan Research Lab
- China HP Software Sales GM
- Alibaba VP Assistant + Public Security Industry Director
- 3x founder; **Haopai reached 300K users with zero paid acquisition**
- Self-funded Volvence R&D with **RMB 5M** since ChatGPT launch
- Hands-on engineering commitment: **GitHub activity since Nov 2022**, DD-verifiable

<!--
I will start with myself, because at this stage you are mostly underwriting founder judgment.

My background has three parts. First, CS and systems training: Peking University CS, IBM Japan Research, and years in enterprise software. Second, sales and market training: HP, Alibaba, Tencent, and large enterprise customers. Third, founder training: I have built products before, including Haopai, which reached 300,000 users with zero paid acquisition.

After ChatGPT launched, I stopped other work and self-funded this direction. The total personal investment has been around RMB 5M, now fully spent on team, engineering, experiments, and JV development.

One thing I want to make concrete: I am not a non-technical CEO outsourcing the product. I have been hands-on in the codebase since late 2022. The GitHub activity is available for DD.

The reason this matters is simple: Volvence sits at the intersection of systems engineering, human relationships, and commercialization. My career prepared me for that intersection, but the next 18 months still have to prove it with real customers.
-->

---

## Team: Scientific Depth + Commercial Access

#### Yang Liu, PhD — Co-founder & Chief Scientist
- CMU PhD; advised by **Avrim Blum and Jaime Carbonell**
- IBM Research, Yale postdoc
- 40+ papers across active learning, drifting distributions, transfer learning
- Full-time

#### Zhao Jiangbo — Founder & CEO
- CS + enterprise software + large-account sales + founder

#### Wang Cangyu — Co-founder / CSO
- Media and private traffic commercialization; TikTok China agency experience

#### Zhang Chi — Co-founder / CTO
- Tsinghua CS; long-time engineering partner

#### Wu Xiang — Co-founder / CMO
- 20 years market and enterprise GTM

<!--
The most important team point is Yang Liu.

Yang is a CMU machine learning PhD, advised by Avrim Blum and Jaime Carbonell. Her research is directly relevant to our problem: active learning, drifting distributions, transfer learning, and learning when the environment is not stationary.

This is not decoration for the deck. Our product problem is: how does an AI system keep learning when each user, each relationship, and each business context changes over time? Yang's research background gives us a serious scientific base for that question.

The rest of the team gives us commercialization access and delivery capability: private traffic, enterprise sales, engineering, and market operations. It is a small team, but everyone is full-time.
-->

---

## The Market Wedge: Private Traffic

> **Private traffic is China's relationship-based commerce layer.**
>
> Public traffic acquisition → WeChat / group / personal account → repeated 1-on-1 relationship → conversion and retention
>
> Existing tools optimize **reach**.
> The unsolved layer is **relationship quality over time**.

```text
Public traffic
    → WeChat add / group join
    → repeated conversation
    → trust / preference / timing
    → purchase / repeat purchase / referral
```

<!--
Our first wedge is private traffic in China.

For an overseas investor, this market can be easy to underestimate. It is not just CRM. It is a relationship-based commerce layer built on WeChat, groups, personal accounts, and repeated 1-on-1 interactions.

Brands and MCNs do not only need to send messages. They need users to feel remembered, understood, and approached at the right time. Existing tools mostly optimize reach: mass messaging, tags, workflows, auto-replies.

The missing layer is relationship quality over time. That is the wedge where our architecture is most directly useful.

Note: do not start with "$150B TAM" or "largest untapped vertical". Start with mechanism. If Patrick believes the mechanism, TAM can follow.
-->

---

## Why Existing Tools Fall Short

**The current stack is built for contact, not relationship.**

| Current private-traffic tools | What customers actually need      |
| ----------------------------- | --------------------------------- |
| Mass messaging                | Remember individual context       |
| Tags and workflows            | Adapt to relationship stage       |
| Keyword automation            | Respond to behavior change        |
| Sales conversion scripts      | Build trust before recommendation |
| CRM records                   | Cross-session relationship state  |

### Our claim: relationship quality becomes a measurable business variable.

<!--
Most tools in this market are useful, but they are built around contact, not relationship.

They help companies reach users, segment users, and automate workflows. But they do not solve the harder question: how should the system behave differently after 20 conversations with the same user?

If a user says, "you are too pushy", the next conversation should reflect that. If a user mentioned a family situation last week, the system should remember it appropriately. If the relationship is not ready for a recommendation, the system should wait.

This is not a better script problem. It requires persistent relationship state, memory, adaptation, and governance. That is why we call Volvence a relationship runtime.
-->

---

## Product: Relationship Runtime

### Thin Prompt. Thick Runtime.

Prompt is only the language rendering layer.

**Runtime owns the product logic:**

- Persistent user and relationship state
- Cross-session memory and preference evolution
- Adaptive behavior policies
- Auditable actions and deletion controls
- Vertical-specific bundles for different business contexts

> Base models are substrate. Prompt renders. **Runtime owns.**

```text
Applications:  private traffic / companion / education / figure
─────────────────────────────────────────────────────────────────
Volvence Relationship Runtime:
   identity · memory · relationship stage · adaptation · governance
─────────────────────────────────────────────────────────────────
Thin prompt:   renders current runtime state into language
─────────────────────────────────────────────────────────────────
Substrate:     GPT / Claude / Qwen / DeepSeek / local models
```

<!--
This is the product definition.

We are not replacing GPT, Claude, Qwen, or DeepSeek. We use them as substrate. The Volvence layer maintains the state that base models do not own: user state, relationship state, adaptation history, and governance.

The simplest way to understand our architecture is: thin prompt, thick runtime.

In a normal LLM wrapper, a lot of product logic is hidden in the prompt: persona, memory summary, behavioral rules, sales posture, safety instructions. That can produce a good demo, but it is fragile. If the prompt changes, the product changes. If the context gets long, behavior drifts. If the customer asks for audit or deletion, you do not have a clean state boundary.

In Volvence, the prompt is only the language rendering layer. The runtime owns the durable logic: identity, memory, relationship stage, constraints, adaptation, and audit trail.

The product question is not "can the model answer this message?" The product question is "does the system know how this relationship has evolved, what it should remember, what it should not say, when to recommend, when to hold back, and how to remain auditable?"
-->

---

## Why This Is The Next Vertical Data Layer

> **A natural extension of "vertical proprietary data > LLM scaling".**

#### Generation 1 vertical data
- Mayo Clinic / S&P / legal / academic / professional corpora
- Existing, recorded, institution-owned
- Powerful, but eventually licensable, synthesizable, or absorbed by frontier models

#### Generation 2 vertical data: **the relationship itself**
- **Real-time:** every interaction creates new data
- **Relational:** user × time × context × relationship stage
- **Non-transferable:** state lives in the owner's runtime snapshots
- **Long-tail by construction:** 100M users = 100M micro-verticals

<hr>

**Implication.** Stronger base models help our substrate. They do not, by themselves, become the persistent relationship owner for any specific brand, MCN, or vertical.

<!--
This is the page where I want to connect our wedge to your thesis.

Xfund's framing — that proprietary vertical data matters more than competing with base-model scaling — is the framing we have used internally for two years. The question we asked is: what is the next layer of vertical data after Mayo Clinic, S&P, and the obvious institutional corpora?

Our answer: the relationship itself.

First-generation vertical data is institutional: medical, financial, legal, academic. It is powerful, but it is already-existing, already-recorded, and over time it becomes licensable or synthesizable.

Second-generation vertical data is the relationship between a brand or a service and an individual user, accumulated over time. Every interaction creates new data. The state is not a record, it is a trajectory: user, time, context, relationship stage. It is non-transferable, because it lives in the owner's runtime snapshots. And it is long-tail by construction, because every user-relationship pair is its own micro-vertical.

I want to be careful here. I am not claiming that a strong base model cannot ship memory features. They will, and that helps our substrate. The claim is narrower: a generic assistant is not structured to be the persistent relationship owner for a specific brand, MCN, or service vertical. That is the layer Volvence is building toward.
-->

---

## Demo Setup: Mobi Private Traffic JV

#### Partner
- **28M-follower MCN**
- Large private traffic pool
- Human operators cannot maintain high-quality 1-on-1 relationships at scale

#### Demo watch points
- Remembers prior context **across sessions**
- Keeps user preferences **separated**
- Recommends only when **relationship stage is appropriate**
- **Adjusts** after user feedback

#### Current status
- JV signed
- Pilot / launch path in progress
- *Conversion uplift not yet proven*

> 🎬 **[ VIDEO — 4-5 min edited Mobi demo ]**

<!--
Before video (1 min):
Let me make this concrete with Mobi. Mobi is a 28M-follower MCN. The business problem is simple: there are too many users and too few human operators to maintain high-quality conversations.

In the demo, please watch four things: cross-session memory, preference separation, recommendation timing, and behavior adjustment after user feedback.

One important caveat before we play it: the JV is signed and the demo is real, but conversion uplift is not yet proven. That is one of the main things this round needs to help us validate.

After video (90s):
The point of the demo is not that the AI sounds human in one turn. Many systems can do that.

The point is that the system has state across turns and sessions. It can remember, adapt, and avoid treating every message as a fresh prompt.

This is the behavior we want to measure in production: not only response quality, but relationship quality and conversion over time.
-->

---

## Why This Is Hard

**Relationship AI is not a prompt problem.**

| LLM wrapper              | Volvence                       |
| ------------------------ | ------------------------------ |
| Prompt owns persona      | Runtime owns identity          |
| Prompt summarizes memory | Runtime owns memory state      |
| Prompt lists rules       | Runtime owns constraints       |
| Prompt nudges behavior   | Runtime owns adaptation policy |
| Prompt is hard to audit  | Runtime produces audit trail   |

### Prompts should render behavior, not own behavior.

**Four runtime capabilities the system must have:**

1. **Identity stability** — the system does not drift randomly across sessions
2. **Memory discipline** — remembers what matters, scopes what should not persist
3. **Adaptation** — behavior changes after feedback and context shifts
4. **Governance** — customers can audit, delete, and constrain behavior

<!--
This is the technical reason we exist.

A single prompt can imitate warmth, empathy, or a persona. But relationship products need stability across time. They need the system to know who the user is, what has changed, what should be remembered, and what should be constrained.

So the technical problem is not "write a better prompt". Our principle is: prompts should render behavior, not own behavior.

In most LLM wrappers, the prompt owns the product logic: persona, memory summary, rules, sales posture, safety constraints. That can work for demos, but it becomes fragile in long relationships.

In Volvence, the durable parts live outside the prompt: relationship state, memory updates, adaptation policy, audit trail, and customer constraints. The prompt should be thin. It expresses the current state into language, not secretly contain the whole product.

This is also why we do not frame ourselves as an agent framework. Agent frameworks coordinate tools. Our problem is maintaining a relationship state over time.
-->

---

## Architecture: Body + Brain, In Plain Terms

#### Body
- Stable identity and constraints
- Needs / drives / restraint parameters
- Relationship posture

#### Brain
- Memory and state update
- Policy and response planning
- Adaptation from feedback
- Auditable action layer

<hr>

> The goal is **not** anthropomorphic theater.
> The goal is **stable behavior over long relationships**.

<!--
Internally we call the architecture Body + Brain.

Body means stable identity and constraints: what kind of entity this is, what it wants to optimize, what it should avoid, and what relationship posture it should maintain.

Brain means the adaptive layer: memory updates, state transitions, response planning, feedback handling, and auditability.

I want to be careful here. We are not selling anthropomorphic theater. We are using this language because long-term behavior needs something like stable drives and adaptive cognition. The business outcome is simple: more stable, more trustworthy relationship behavior over time.
-->

---

## Scientific Base: Learning Under Change

#### Relationship products face changing distributions
- Users change preferences
- Relationship stages evolve
- Business goals shift
- Feedback is sparse and noisy

#### Yang Liu's research base
- Active learning
- Drifting distributions
- Transfer learning
- Nonstationary learning

> This gives Volvence a serious starting point for **data-efficient adaptation**.

<!--
This is where Yang's research background becomes directly relevant.

In relationship products, the environment is not stationary. A user's preference changes. A relationship stage changes. Business context changes. Feedback is sparse, delayed, and noisy.

Yang has worked for years on active learning, drifting distributions, transfer learning, and nonstationary learning. That does not magically solve the product problem, but it gives us a serious scientific base for building systems that adapt with limited data.

I would frame our current evidence as early but real: we have engineering tests for memory, persistence, and multi-timescale adaptation. During DD, we can show the repo, test surfaces, and what has or has not been proven.

Caution: do not say "O(log n) means we only need 1/1000 data" in main deck. If asked, discuss active learning theory carefully as a scientific basis, not a direct business guarantee. (See Appendix C.)
-->

---

<!-- _class: dense -->

## Evidence We Can Show Today

**Technical evidence, in plain English** — we are not asking you to believe a diagram. We can show the system doing five hard things:

| Hard thing                    | Plain-language test                                                                    | Why it matters                                         |
| ----------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Remembers across sessions** | User says something in session 1; after restart, system uses it correctly in session 2 | This is not a long prompt pretending to remember       |
| **Updates old beliefs**       | User corrects prior info; system stops using the outdated version                      | RAG retrieves; relationship systems must revise        |
| **Keeps users separated**     | Alice's preferences never leak into Bob's behavior                                     | Required for any enterprise/private-traffic deployment |
| **Can test changes safely**   | New behavior runs in SHADOW before becoming ACTIVE                                     | Customers need reliability, not demo magic             |
| **Runs with thin prompts**    | Core behavior comes from runtime state, not a giant hidden prompt                      | The product is architecture, not prompt engineering    |

#### Commercial evidence
- **6** signed JV agreements · **45M+** follower connection base · **50K+** enterprise customer connection base

#### Open gaps
- **0** real ARR from JVs as of this deck
- Conversion uplift **not yet proven**

<!--
This is the evidence page, and I want to make the technical part concrete.

We are not asking you to believe a beautiful architecture diagram. We can show the system doing five hard things that normal LLM wrappers struggle with.

First, it remembers across sessions. A user says something in session one; after restart, the system can still use it correctly in session two. A long prompt can look like memory inside one conversation, but it does not prove relationship continuity.

Second, it updates old beliefs. If a user corrects information, the system should stop using the outdated version. This is a bigger deal than retrieval. RAG can retrieve old notes; a relationship system has to revise what it believes.

Third, it keeps users separated. Alice's preferences cannot leak into Bob's behavior. That sounds basic, but it is essential for enterprise and private-traffic deployments.

Fourth, we can test behavior changes safely. New modules can run in shadow before becoming active, so we can compare old behavior and new behavior before customers depend on it.

Fifth, the prompt is thin. Core behavior is not stored in a giant hidden system prompt. The durable logic lives in runtime state. The prompt mostly renders that state into language.

On the commercial side, we have 6 signed JVs and access to a large audience and enterprise base. But the open gaps are equally important: the JVs have not yet produced real ARR, and conversion uplift is not yet proven.

So the company is not "proven". It is at the point where the next 6-18 months can turn technical proof into business proof.

Optional live demo flow if Patrick asks for proof instead of slides (3 min):
1. Create a user memory in session 1.
2. Restart or reload the runtime.
3. Ask a related question in session 2.
4. Correct one fact and show the system stops using the old version.
5. Switch user identity and show no preference leakage.

The point is not that the answer is poetic. The point is that the runtime state behaves correctly.
-->

---

## Traction: 6 JVs, One Validation Plan

**From signed access to real ARR**

#### Current
- **6 signed JVs**
- 3 private-traffic related
- 3 other verticals: companion, parenting, cross-border commerce / enterprise

#### 18-month validation plan

| Timeline | Milestone                      | Success criteria                                        |
| -------- | ------------------------------ | ------------------------------------------------------- |
| M0-M3    | First lighthouse in production | Real users, real usage, measurable retention/conversion |
| M4-M9    | 3 JVs in production            | Repeatable deployment process, early revenue            |
| M10-M18  | **ARR > $1M real**             | Not projected; **recognized revenue**                   |

> If we do not hit these, **we should not raise a Series A**.

<!--
The main commercial question is whether signed access becomes real revenue.

We have 6 JVs. That is not the same as ARR. I want to be explicit about that. Our 18-month plan is to turn at least 3 of them into in-production deployments and reach more than $1M real ARR.

The first three months should produce the first lighthouse: real users, real usage, and measurable retention or conversion. By month nine, we should have three JVs in production. By month eighteen, we should have real ARR, not projected ARR.

If we do not hit those milestones, we should not be raising a Series A on this story.

Note: this replaces the big 2026-2028 financial projection as the main commercial claim. Detailed scenario lives in Appendix D.
-->

---

<!-- _class: dense -->

## Business Model: Mobi Unit Economics + Kill Criterion

#### Mobi private-traffic JV — signed revenue structure (per converted user, per year)

| Item                            | Unit                                       | 2026 target scale | Volvence revenue contribution |
| ------------------------------- | ------------------------------------------ | ----------------- | ----------------------------- |
| Service fee / token procurement | RMB 30 / user / year                       | ~187K orders      | ~RMB 2.8M                     |
| JV profit share                 | RMB 100 / user / year distributable profit | same              | ~RMB 2.8M                     |
| **Mobi JV 2026 subtotal**       |                                            |                   | **~RMB 5.6M (~$800K)**        |

#### Conversion assumption <span class="projected">[ projected, not proven ]</span>
- Industry baseline (SCRM, public reporting): **~0.3%**
- Volvence projected with relationship runtime: **0.6 – 1.0%**
- Pilot data not yet through full 3-month observation window
- <span class="kill">KILL CRITERION: if 3-month pilot conversion &lt; 0.5%, this vertical is deprioritized</span>

#### Repricing thesis

| Weimob / Youzan         | Volvence                            |
| ----------------------- | ----------------------------------- |
| Reach tools             | Relationship engineering            |
| Broadcast / auto-reply  | Remembered and understood           |
| One-time conversion     | Cross-session LTV                   |
| ~RMB 1K / month / brand | **RMB 5K – 50K / month / brand** (target) |

<!--
This is the page where I want to be specific about money, and equally specific about what would make us walk away.

The Mobi JV has two revenue layers per converted user per year: a service fee of about RMB 30 that covers our platform and token procurement, and a profit share of about RMB 100 on the distributable margin. At a 2026 target of around 187,000 converted orders, that contributes roughly RMB 5.6M, about $800K, to Volvence from this single JV.

I have to be explicit about the conversion assumption. The industry baseline for SCRM-style private traffic is around 0.3%. Our projected uplift with the relationship runtime is 0.6% to 1.0%. That projection is anchored on the demo behavior — recommendation timing, preference separation, repair-after-rupture — but the 3-month pilot observation window is not yet complete. So this is a projection, not a result.

The kill criterion is the part I want you to remember. If the 3-month pilot lands below 0.5%, we deprioritize this vertical. We do not double down to defend the thesis. That discipline applies to every JV in the portfolio.

The underlying commercial thesis is not "we are a cheaper Weimob". It is repricing. Weimob and Youzan sell reach, so their ARPU is around RMB 1K per brand per month. We are selling long-term relationship optimization, which gives us a path to RMB 5K-50K per brand per month. That is where the multiple comes from, if the conversion thesis holds.
-->

---

<!-- _class: dense -->

## Global Expansion: Public → Private → Platform

**The platform opportunity:** public audience → private relationship → supply orchestration

China private traffic is the first wedge because the behavior already exists at scale.
The same pattern exists globally:

| Layer                         | China wedge                           | Global analogues                                       |
| ----------------------------- | ------------------------------------- | ------------------------------------------------------ |
| **Public domain**             | Douyin / Xiaohongshu / public traffic | TikTok / Instagram / YouTube / podcasts / newsletters  |
| **Private relationship**      | WeChat / groups / 1-on-1              | WhatsApp / Discord / SMS / email / creator communities |
| **Relationship intelligence** | Long-term user state and trust curve  | Preferences, intent, timing, life context              |
| **Supply orchestration**      | JV partners and vertical supply       | Brands, creators, service providers, local commerce    |

> TikTok understands **attention** through content behavior.
> Volvence's opportunity is to understand **customers** through relationship behavior.

**Long-term option:** a relationship-commerce platform, not just a SaaS tool.

<!--
I want to add one strategic point, because it explains why this can expand beyond China.

China private traffic is our first wedge because the behavior already exists at massive scale. Public traffic is pulled into WeChat, groups, and 1-on-1 relationships. Commerce then happens through repeated trust, not one ad click.

But this pattern is not uniquely Chinese. Globally, public audiences also move into private or semi-private channels: TikTok creators move fans into Discord, WhatsApp, SMS, email lists, communities, memberships, and commerce flows.

The missing layer is the same: once a user leaves the public algorithmic feed and enters a private relationship channel, who understands that user over time? Who remembers their preferences, intent, life context, trust stage, and timing?

TikTok understands attention through content behavior. Volvence's opportunity is to understand customers through relationship behavior.

That creates a platform option. I would still frame this carefully: this is not the next 12-month proof. The next 12-month proof is production usage and real ARR. But it is the reason the wedge can become much larger if the runtime works.
-->

---

<!-- _class: dense -->

## Risks

**Main risks and how we test them**

| Risk                                                   | What would prove / disprove it                                                          |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| JV access does not convert to usage                    | First lighthouse fails to retain users or produce conversion signal                     |
| Relationship quality does not improve business metrics | A/B or cohort data shows no uplift over SCRM baseline                                   |
| Base models add stronger memory                        | We prove value in vertical state, governance, and business workflow, not generic memory |
| Regulatory / safety constraints tighten                | Audit, deletion, consent, and scoped memory become mandatory product features           |
| Team execution bandwidth                               | 3 in-production JVs by M9, otherwise narrow focus                                       |

> The next round is about **converting these risks into evidence**.

<!--
These are the risks I would focus on if I were in your seat.

First, signed JVs may not convert to real usage. Second, relationship quality may not produce business uplift. Third, base models may add stronger memory. Fourth, regulation may tighten. Fifth, a small team may be stretched too thin.

Our answer is not to hand-wave these risks. It is to test them quickly. If we cannot get lighthouse usage, retention, conversion signal, and three in-production deployments, then the story is not ready for Series A.

On substitution risk: if base models get better memory, that helps the substrate. Our value has to be vertical relationship state, governance, and workflow integration. That is the proof we need to produce.
-->

---

<!-- _class: ask -->

## The Ask — Late Seed / Pre-Series A

| Dimension                      | Target                                           |
| ------------------------------ | ------------------------------------------------ |
| Round size                     | **$3M – $5M USD**                                |
| Pre-money valuation (range)    | **$20M – $30M USD** *(under discussion)*         |
| Xfund target ticket            | **$1.5M – $2.5M**, lead or co-lead               |
| Equity to Xfund                | **~7 – 10%**                                     |
| Runway                         | **18 months**                                    |

#### Use of funds
- **Engineering 40%** — relationship runtime, deployment reliability, evaluation
- **Compute / data 25%** — substrate, evidence pipeline, benchmark runs
- **GTM 20%** — 3 in-production JV launches, lighthouse customers, partner success
- **Operations / legal / IP 15%** — audit, consent, deletion, IP structure

#### Next financing gate
**3 JVs in production** · **ARR > $1M real** · **Repeatable deployment playbook**

<!--
We are raising $3M to $5M as a late seed or pre-Series A round, with Xfund as lead or co-lead if there is alignment.

On valuation, the target range we are working with is $20M to $30M pre-money. I want to flag that as a range under discussion, not a take-it-or-leave-it number. We would expect Xfund's ticket in the $1.5M to $2.5M band, which corresponds to roughly 7% to 10% equity at that range.

The purpose of this round is not broad expansion. It is validation. We need 18 months to turn signed access into production deployments, real revenue, and repeatable evidence.

The next financing gate is clear: 3 JVs in production, more than $1M real ARR, and a repeatable deployment playbook.

If we hit that, we can raise Series A from a position of evidence. If we do not, we should narrow or rethink the wedge.
-->

---

<!-- _class: divider -->

# The thesis

<br>

AI products that matter over time need **relationship infrastructure**.

Human relationships are the **next vertical proprietary data layer**.

The first wedge is **private traffic**.

<br>

The next 18 months are about proof:
**signed access → production usage → real ARR**

<!--
The thesis is simple.

Many AI products will not be judged by one answer. They will be judged by whether they can maintain a useful, trusted relationship over time.

Volvence is building the runtime for that. Relationship data — the second-generation vertical data — is what accumulates inside that runtime. Private traffic is our first wedge because the market already has relationship demand and distribution access.

The next 18 months are about proof: signed access to production usage to real ARR.

I would love to spend the rest of the time on your questions.
-->

---

<!-- _class: divider -->

# Appendix

<!--
The following slides are backup. Use only when Patrick asks.
- A: Companion Benchmark — "How do you evaluate relationship quality?"
- B: Einstein / Figure Bundle — "Is this only for private traffic?" / "How do I know this is not just a wrapper?"
- C: Active Learning — "How does Yang's active learning work connect to the product?"
- D: 3-Year Financial Scenario — "What could this become if the JV model works?"
- E: Technical Proof Ladder — "How do I know this is not just a wrapper?"
-->

---

## Appendix A — Companion Benchmark

**Goal:** evaluate long-session relationship behavior, not single-turn preference.

#### Current design
- 24 public scenarios
- 96 held-out private scenarios
- 30-turn arcs
- Axes: relationship continuity, adaptation, boundary maintenance, temporal fidelity, theory of mind, regime stability

#### Current status
- Benchmark design exists, **open-sourced under Apache 2.0**
- Reference SUT runs not complete
- Judge robustness still under validation

<!--
We built Companion Benchmark because existing benchmarks mostly measure single-turn preference, task completion, or role-play consistency. Our category needs long-session relationship evaluation.

But I would not overclaim this today. The benchmark is designed, but the reference model runs and judge robustness are still in progress. It is a useful evaluation asset, not yet an industry standard.
-->

---

<!-- _class: dense -->

## Appendix B — Einstein / Figure Bundle

**Shipped artifact:** `figure-bundle:einstein:29eacd226a7cdfd0` *(immutable, reproducible, auditable)*

#### Four fidelity layers

| Layer       | What it means                      | Architecture evidence       |
| ----------- | ---------------------------------- | --------------------------- |
| L1 Voice    | Sounds like him                    | Body / style prior          |
| L2 Position | Agrees on topics he wrote about    | Soul Migration              |
| L3 Citation | Substantive claims trace to source | Memory + grounded decoder   |
| L4 Refusal  | Refuses outside documented scope   | ScopeRefuser + coverage map |

#### Strategic use
- Demonstrates **auditability and scope control**
- Relevant to museums, education, publishing, IP holders
- Not the first commercial wedge

<!--
Einstein is a useful case study because it shows the runtime's auditability and scope control. A normal persona prompt tries to answer everything. Our figure bundle can refuse when outside documented material.

The L4 refusal layer is what matters most. Museums, universities, and publishers do not buy theatrical realism — they buy governed fidelity.

That said, I would not make this the center of this fundraising story. The first wedge is private traffic because it has clearer distribution and revenue path.
-->

---

## Appendix C — Active Learning Claim, Carefully Stated

#### Careful claim
Active learning theory shows that, **under certain assumptions**, selected labels can reduce sample complexity versus passive labeling.

> Reference: Hanneke & Yang, *Minimax Analysis of Active Learning*, JMLR 2015.

#### Product relevance
- Relationship data is sparse and noisy
- Feedback is expensive
- Vertical partners do not have internet-scale labeled datasets

#### What we still need
- Internal task-level ablations
- Passive vs active learning curves
- Production feedback loops

<!--
I would state this carefully.

Yang's work gives us a serious foundation for data-efficient learning under changing conditions. In theory, active learning can substantially reduce label complexity under specific assumptions.

But I would not claim that theory directly guarantees a 1/1000 data advantage in every vertical. What we need to show in DD and production are task-level ablations: passive versus active learning curves, label efficiency, and product metrics.
-->

---

## Appendix D — 3-Year Financial Scenario

> **Scenario, not proof.**

| Metric               | 2026    | 2027     | 2028     |
| -------------------- | ------- | -------- | -------- |
| Revenue RMB          | 35M     | 165M     | 408M     |
| USD equivalent       | ~$5M    | ~$23.6M  | ~$58.3M  |
| Net margin           | 31%     | 46%      | 54%      |
| Project gross margin | 55%     | 65%      | 75%      |

#### Drivers
- 3-6 JVs entering production
- Mobi-style unit economics repeating across verticals
- Substrate cost amortization
- Fixed-cost dilution

<span class="projected">All figures should be treated as projections until real ARR is recognized.</span>

<!--
We have a detailed internal model, and we can share it during DD.

The number to anchor on is 2026 revenue of around RMB 35M, or roughly $5M. The 2027 and 2028 figures depend on whether the unit economics from Mobi repeat across other verticals.

The point of this table is not "we will do $58M in 2028". The point is that the projection has anchors: signed partner audience size, industry baseline conversion, and the contracted revenue share structures. I would still treat all of it as scenario planning, not proof. The proof points are production deployments, conversion signal, and recognized ARR.
-->

---

<!-- _class: dense -->

## Appendix E — Technical Proof Ladder

**Thin Prompt, Thick Runtime**

| Level | What we prove                                      | Investor translation                                           |
| ----- | -------------------------------------------------- | -------------------------------------------------------------- |
| L1    | Same user, same relationship state across sessions | It can remember beyond one chat                                |
| L2    | Corrections update future behavior                 | It can learn, not just retrieve                                |
| L3    | Different users stay isolated                      | It is deployable in real customer environments                 |
| L4    | New behavior can run SHADOW before ACTIVE          | It can improve without breaking production                     |
| L5    | Core behavior works with thin prompts              | Prompt renders; runtime owns                                   |
| L6    | Metrics compare old vs new behavior                | It can become an engineering discipline, not founder intuition |

> Identity, memory, relationship stage, constraints, adaptation, and audit live in the **runtime**.
> The prompt is only the final language interface.

<!--
If someone asks whether this is just a wrapper, I would answer with this ladder.

L1: same user's relationship state across sessions. Minimum requirement.
L2: revise old information after correction. The difference between retrieval and learning.
L3: user states stay isolated. Required for real deployment.
L4: new behavior runs in shadow before it affects customers. A safe way to improve.
L5: core behavior works with thin prompts. The point is not that there is literally no prompt anywhere; every LLM call has some interface instruction. The point is that the prompt does not own the product logic — it renders runtime state.
L6: compare old and new behavior with metrics. This is when the system becomes an engineering discipline rather than founder intuition.

The technical moat is not one magical model or one clever prompt. It is the runtime discipline around relationship state.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Thank you.

### Volvence — Thin Prompt. Thick Runtime.

