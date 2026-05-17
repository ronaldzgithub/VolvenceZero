# VolvenceZero — Xfund Pitch Deck v8

> Status: **v8.0 draft (2026-05-17)**
> Spine: **Relationship Continuity Creates Economic Lift.** Everything in the body deck serves one empirical question — does persistent relationship state move retention, conversion, and LTV? Civilization-level narrative (AGI, cognitive AGI, token-RL impossibility, "frontier labs structurally cannot") is removed from the body and preserved only in a senior-technical-DD appendix.
>
> Tone: parallel-ecosystem, not gladiator. We do not say "OpenAI cannot do X." We say "**frontier labs optimize for generality; we optimize for persistent relationship state.**" That is mature, not religious.
>
> Audience: Patrick Chung first; his technical DD partner second.
>
> Recommended meeting format: **~35 min presentation + ~25 min conversation**. Short version: Slides 1-3, 5, 8-11. Demo + Mobi unit economics is the centerpiece.

---

## V8 vs V7: what changed and why

V7 still carried part of an AGI-class manifesto in the body (Cognitive AGI as Step 1, "frontier labs structurally locked out" as Step 5). External read on the v7 draft was sharp and correct: that framing collapses if a frontier lab ships memory + persona + state next quarter, and it pattern-matches to "founder building optimal-strategy on what a giant *will not* do." A 25-year venture career will read that as fragile.

V8 reorganizes around an **empirical, falsifiable, business-metric spine**:

1. **Spine sentence becomes "Relationship Continuity Creates Economic Lift."** Everything in the body serves this. Retention, conversion, LTV, persistent state, cohort math — these are the words Patrick can carry into an LP room.
2. **AGI narrative, frontier-lab structural arguments, and token-RL papers are removed from the body** and moved to Appendix E (senior-technical-DD only). Civilization-level claims that we cannot validate in 6 months are not in the room when commercial validation has not yet shipped.
3. **The "OpenAI structurally cannot ship X" framing is replaced everywhere** with "**frontier labs optimize for generality; we optimize for relationship state**." Parallel ecosystem, not gladiator match.
4. **Mobi becomes the centerpiece — three slides instead of two.** Concrete unit economics drawn from the internal financial model (`docs/business/00：19-详细版本(1).xls`): 28M follower pool, **0.67% conversion**, 187K orders in 2026, RMB 30/user/year service fee, RMB 100/user/year profit share, **RMB 5.6M (~US$800K) Volvence revenue from this single JV in 2026**, growing 3× / 6× in 2027 / 2028. Plus a before/after conversation pattern showing the four runtime properties as four LTV levers.
5. **A new "How we will measure this in 12 months" slide.** Honest: we do not yet have a one-year retention curve. We commit to the four cohort metrics we will produce, with kill criterion.
6. **The Generation-2-vertical-data spine slide is kept** (this was the soul of v7 per external read) but reframed less philosophically. Bloomberg / Visa / Salesforce / TikTok are explicit analogues: interaction itself becomes moat.
7. **Yang Liu reframed as "small-data continual learning at production scale" enabler**, not as proof-of-cognitive-AGI theorem. Her math is the *reason* we believe a few hundred turns of per-user data is enough — that is a business-metric claim.
8. **Governance-as-moat is kept** but trimmed of AGI references. Five regulatory pressures × five shipped engineering pieces.
9. **`5 vertical lifeforms in one process`, `ModificationGate`, `R10/R14`, `β_t / z_t / metacontroller`, `Nested Learning`, `ETA`, `R1-R15`** — all removed from the body. Either dropped or relegated to Appendix E.
10. **The v7 8-step technical thesis is preserved verbatim in Appendix E** for the technical DD reviewer Patrick will hand the deck to.

---

## Meeting cadence

```text
0-2 min      Cover + one-line pitch
2-6 min      Founders (Hacker + Humanist + Yang Liu as stabilizer)
6-10 min     Generation 2 vertical proprietary data            ← deck spine
10-16 min    Relationship continuity → economic lift mechanism
16-22 min    Governance is the third moat
22-30 min    MOBI — partner, conversation, unit economics      ← centerpiece
30-37 min    Distribution, 2026 ARR, 18-month proof plan
37-42 min    Technical credibility (one slide), risks, ask
42-60 min    Conversation
```

If Patrick interrupts, switch to conversation. Every block stands alone.

---

# Main Deck

## Slide 1 — Cover

**On-screen**

> **VOLVENCE**
>
> **Relationship Continuity is the next data moat.**
>
> Mayo Clinic data was the first layer of vertical proprietary data.
> The per-person trajectory — every conversation, every preference, every adaptation, accumulated over months — is the next.
> Renewable. Non-transferable. Auditable. Owned by no foundation lab.
>
> Zhao Jiangbo, Founder & CEO
> Xfund conversation · May 2026

**Speaker script (60s)**

Patrick, thank you for the time.

Here is the thesis in one sentence. Mayo Clinic data is the first layer of vertical proprietary data — Open Evidence is the proof. The next layer is the per-person trajectory each user produces with a product over months and years. **The question we are trying to answer empirically is whether sustaining that trajectory — remembering the user, scoping preferences across users, timing recommendations correctly, adapting persistently after feedback — measurably lifts retention, conversion, and LTV.**

If the answer is yes, the trajectory becomes a new category of vertical proprietary data. We have the team, the engineering, and the distribution — 45M followers and 50K enterprises through 6 signed JVs — to measure this in the next 12 months.

Six signed JVs. 2026 conservative ARR US$3.3M-5M. The single largest JV — Mobi private-traffic — we will walk through in detail.

**Design note**

Cover should be near-black. Only one phrase visible at first frame: **"Relationship Continuity is the next data moat."** Everything else fades in.

---

## Slide 2 — Founders: Hacker + Humanist, with a CMU-trained stabilizer

**On-screen**

> **Two principals; one team thesis: Hacker + Humanist.**
>
> **Zhao Jiangbo — Founder & CEO** (full-time)
>
> - PKU CS · MBA — *Hacker side*: GitHub commits since Nov 2022, DD-verifiable
> - IBM Japan Research · HP China Software Sales GM · Alibaba VP Office · Tencent industry director — *Humanist side*: built and sold enterprise narratives across three of the largest tech employers in Asia
> - 3× founder; Haopai 300K users with zero paid acquisition; commercial exit
> - Self-funded Volvence R&D with RMB 5M since ChatGPT launch
>
> **Yang Liu, PhD — Co-founder & Chief Scientist** (full-time)
>
> - CMU PhD, advised by **Avrim Blum** and **Jaime Carbonell** (founding director, CMU Language Technologies Institute)
> - IBM Research · Yale postdoc
> - **40+ peer-reviewed papers** — 18 A-tier (NeurIPS / ICML / JMLR / COLT / SODA / FOCS / AISTATS); upcoming NeurIPS 2026
> - Field: **how to learn fast from very little data when the underlying distribution is changing** — exactly the math this product needs
>
> **Wang Cangyu, CSO** · PhD Psychology · ex-Zhongqi Media GM (Douyin best agency, RMB 6B revenue)
> **Zhang Chi, CTO** (full-time) · Tsinghua CS · ex-Glodon · Haopai co-founder
> **Wu Xiang, CMO** (full-time) · ex-HP, Neusoft executive · 20yr enterprise GTM

**Speaker script (3 min)**

The Xfund framing since 2012 has been Hacker + Humanist — the lateral thinker who is both a builder and someone who can think in long sentences. I want to show why this team profile fits.

I am the hacker-humanist. PKU CS, then MBA, then product / sales / strategy at IBM Japan Research, HP China, Alibaba, and Tencent. Three founder cycles. Haopai grew to 300K users with no paid acquisition; we sold it commercially. I write code on Volvence today — my GitHub commits since November 2022 are public. I am not a non-technical CEO outsourcing engineering.

Yang Liu is the technical stabilizer. CMU PhD under Avrim Blum and Jaime Carbonell. Forty peer-reviewed papers — 18 at A-tier venues. The field she has spent 15 years building is **the math of learning fast from very little data, when the user changes over time**. That is precisely the regime real consumer relationships produce — a few hundred turns a month per user, with the user's needs drifting as life changes. We did not build a company and look for a scientist; the science came first and the company sits on it.

Wang Cangyu — psychology PhD plus GM of Zhongqi Media, the agency Douyin officially recognizes as best, with RMB 6B in annual revenue. Zhang Chi — Tsinghua CS, ex-Glodon, my Haopai co-founder. Wu Xiang — 20 years enterprise GTM through HP and Neusoft.

That is the team you are underwriting.

**Design note**

Yang's CMU advisors and paper count are the credibility anchors. Make them visually heavier. "Hacker + Humanist" is a header.

---

## Slide 3 — The deck spine: Generation 2 vertical proprietary data

**On-screen**

> **Three generations of "vertical proprietary data beats LLM scaling".**
>
> ```
> Generation 0 — Public-internet text          (closing ceiling 2026-2032)
>
> Generation 1 — Institutional vertical data
>   ↳ Mayo Clinic × Open Evidence              (Xfund portfolio; finite; licensable)
>
> Generation 2 — Per-person trajectory data
>   ↳ THIS IS VOLVENCE                         (renewable; non-transferable; auditable)
>
> Adjacent — Snapshot replica of one person
>   ↳ Delphi                                   (Xfund portfolio; static clone)
> ```
>
> **The closest precedents for "interaction itself becomes moat":**
>
> | Company | Asset | Why competitors cannot copy |
> |---|---|---|
> | Bloomberg | Decades of co-produced market interaction with terminal users | Switching cost is the embedded workflow, not the data |
> | Visa | The transaction graph between merchants and consumers | The graph is the product; new entrants cannot retroactively produce it |
> | Salesforce | The CRM record co-produced by each enterprise with its customers | Migration cost grows with relationship age |
> | TikTok | Per-user interaction graph driving recommendation | The graph is non-transferable; ByteDance cannot give it to a competitor even if forced to |
>
> **Generation 2 is the same pattern, applied to AI**: the trajectory is jointly produced with the user, compounds across sessions only if a thick runtime exists to remember / scope / govern it, and is non-transferable because nobody else has the consent or the history.

**Speaker script (3 min)**

The most important slide in this deck, and I want to frame it inside your own portfolio.

Open Evidence is the canonical Generation 1 proof. Mayo Clinic data is enormously valuable, and it is also a particular kind of data — institutional, recorded, ultimately licensable. There is a finite list of Mayo-Clinic-equivalents in the world.

Delphi sits next to that, capturing one person statically. A snapshot replica.

Volvence sits one layer past both, in Generation 2: **the live trajectory each user produces with a product**, accumulating across sessions, jointly produced with them, owned by neither party alone, compounding only if a runtime exists to remember and govern it.

The closest precedents are not other AI companies. They are Bloomberg, Visa, Salesforce, TikTok. In each case the underlying asset is *interaction itself* — the workflow with terminal users, the transaction graph between merchants and consumers, the CRM record co-produced with each customer, the per-user recommendation graph. Each of those assets is non-transferable, renewable, and compounds with time. New entrants cannot retroactively produce them.

Generation 2 is the same pattern applied to AI. If this thesis is right, the question Volvence is building to answer is **whether sustaining that trajectory — through persistent memory, persona scoping, recommendation timing, and adaptation after feedback — produces measurable economic lift over a baseline that does not sustain it.** That question gets answered on real partners in real verticals. Mobi is the first answer; the next 18 months produce the data.

**Design note**

The three-row generation diagram is the visual lede of the entire deck. The four-row analogue table is the supporting structure. Spend layout time on this slide.

---

# Part A — Why relationship continuity is economic

> Three slides. Each one is delivered in the vocabulary of retention, conversion, LTV, and governance — the metrics that decide whether Generation 2 is a real category or a thesis.

---

## Slide 4 — The data ceiling is closing on text, not on people

**Claim**

> Public-internet text exhausts between 2026 and 2032 by published estimates. Institutional vertical data extends the runway but is enumerable. **The only data layer that scales with the number of users × the number of days they show up is per-person trajectory.** That layer is renewable for the same reason every user is renewable — they come back tomorrow.

**On-screen**

> | Data layer | Volume | Renewable? | Locked-in to operator? |
> |---|---|---|---|
> | Public-internet text | ~10-50 TB indexed | No — exhausted 2026-2032 (Epoch AI) | No |
> | Institutional vertical (Mayo, Bloomberg, Westlaw) | ~100s of TB per institution | No — finite per institution | Per license |
> | **Per-person trajectory at 1B users × 10 MB/day** | **~3.6 EB / year** | **Yes — daily** | **Yes — joint with user** |
>
> Magnitude ratio between row 1 and row 3: **~10⁵**.
>
> **What the labs are doing about it (independent market signal):**
>
> - OpenAI's 2024-2026 product pivot toward memory, persistent users, consumer integration
> - Anthropic's 2024-2026 pivot toward coding and stateful tools
>
> Both labs are racing toward the same layer. **They will optimize for generality across one billion users. We optimize for relationship state inside one vertical.** Both can coexist; both produce different shapes of moat.

**Speaker script (2 min)**

Generation 2 is structural, not speculative. Epoch AI's 2024 paper put a number on the text ceiling — exhausts somewhere between 2026 and 2032. After that, scaling by reading more of the internet does not work.

Open Evidence's Mayo deal moved Generation 1 forward. The math on Generation 2 is different by five orders of magnitude. A modal smartphone user produces around ten megabytes of contextual data a day. A billion users for a year is ~3.6 exabytes; the entire indexed text web is ~50 terabytes. The ratio is roughly ten to the fifth.

OpenAI and Anthropic both see this picture. Both are pivoting toward memory, persistent users, and stateful workflows. **They will optimize for generality at billion-user scale; we optimize for relationship state inside one vertical.** Those are different moats; both can be real.

The question is whether maintaining the trajectory inside a vertical actually moves business metrics. That is what the next slide is about.

**Design note**

The 10⁵ ratio is the slide. Make it the visual anchor. The final paragraph in the box — "they optimize for generality; we optimize for relationship state" — is the line we want Patrick to remember.

---

## Slide 5 — Relationship Continuity → Economic Lift (the new spine)

**Claim**

> Four runtime properties translate directly to four levers on the LTV function. We do not yet have a year-long retention curve to prove the magnitudes; we have a pilot on a real partner with the math to measure them and a kill criterion if they do not hold.

**On-screen — four runtime properties × four LTV levers**

> | Runtime property | What changes for the user | LTV lever | How we will measure on Mobi |
> |---|---|---|---|
> | **Cross-session memory** | User does not re-explain themselves on day 30 | **Retention curve** — D30/D90/D180 cohort retention vs. SCRM baseline | Cohort retention curve, Mobi pilot, M0-M6 |
> | **Preference separation** | Alice's preferences never leak into Bob's responses | **Trust premium** — reduced unsubscribe, higher willingness to pay | Unsubscribe rate, paid upgrade rate vs. baseline |
> | **Recommendation timing** | System holds back when relationship stage is wrong; recommends when it is right | **Conversion uplift** — 0.6%-1.0% target vs. SCRM baseline 0.3% | Conversion rate vs. control cohort, weekly cohorts |
> | **Adaptation persistence** | User pushback ("you are too pushy") changes next session and survives restart | **Recovery rate** — drop in churn after a typed-feedback event | Post-rupture cohort retention, repair audit log |
>
> **Honest disclosure (read first):**
>
> - We have **not** yet shipped a 6-month retention curve.
> - We **have** shipped the runtime that produces persistent state, plus reproducible memory probes (4/4 PASS in repo where baseline retrieval fails on 3/4).
> - The Mobi pilot is the first commercial measurement. The kill criterion (Slide 10) is the public discipline.
> - **If by M6 we cannot show a measurable lift on at least two of the four levers above, the relationship-continuity thesis softens and we narrow.**

**Speaker script (4 min)**

This is the slide I most want you to read carefully, because it is what the round is for.

The bet is not "AI sounds more empathetic." Every base model can sound empathetic for one turn. The bet is that **four specific runtime properties move four specific business metrics**, in measurable ways, over months.

Cross-session memory moves the retention curve. If a user does not have to re-explain themselves on day 30, day 30 retention goes up. Mobi pilot will measure D30 / D90 / D180 against the SCRM baseline cohort.

Preference separation moves trust premium. If Alice's preferences never appear in Bob's responses, unsubscribe goes down and willingness to pay goes up. Same pilot measures both.

Recommendation timing moves conversion. SCRM industry baseline is ~0.3% conversion. Our target is 0.6 to 1.0%. The hypothesis is that holding back when the relationship stage is wrong, and recommending when it is right, doubles or triples conversion.

Adaptation persistence moves recovery. When a user explicitly says "you are too pushy," that becomes a typed event, durable memory, and the next session behaves differently — and stays different. The metric is post-rupture retention.

I want to be clear about what we do not have. We do not yet have a six-month retention curve, because we just shipped. We do have the runtime that produces the persistent state, and we have the memory probes in the repo passing four out of four where baseline retrieval fails three of four. **The point of the round is to convert those four mechanisms into four cohort curves on Mobi.**

If by month six we cannot show measurable lift on at least two of those four levers, the relationship-continuity thesis softens and we narrow. That is the kill criterion at the thesis level.

**Design note**

This is the new spine of the deck. Four-row table is the visual; the "honest disclosure" box is bordered in green and unmissable. Patrick will read the honest disclosure first — that is intentional.

---

## Slide 6 — Governance is the third moat, and it is the moat Xfund underwrites well

**Claim**

> The runtime architecture this product requires — owner-snapshot SSOT, audit trail, scoped deletion with evidence, consent-bound memory, gated self-modification — is the same architecture EU AI Act, GDPR Article 17, and China PIPL are about to mandate for any product touching per-person behavioral data. **We built that architecture before regulation forced it.**

**On-screen**

> | Regulatory pressure | What it requires | What we shipped |
> |---|---|---|
> | EU AI Act (high-risk per-user systems, 2026-2027 enforcement) | Audit trail of automated decisions, training-data transparency, scoped purpose | Immutable snapshot SSOT with publish-time provenance |
> | GDPR Article 17 right to be forgotten | Per-user deletion with evidence | `DELETE /v1/users/me/memory` + deletion-evidence ledger |
> | China PIPL cross-border data | Per-user scoped storage and consent | Per-user memory partitioning; consent surfaces in API |
> | AI liability for autonomous self-modification | A gate that decides when the AI is allowed to change itself | Modification-gate primitive with rollback evidence |
> | Per-persona refusal (enterprise audit requirement) | Refusal at persona level, not at model level | Per-persona / per-figure refusal already implemented |
>
> **The same architecture is the legal basis for enterprise contracts.** Our 50K enterprise channel through 1688 / Hengyi cannot sign without these properties. A corporate audit counsel cannot put GPT-with-Memory on a regulated workflow today; they can put Volvence on it.
>
> *Note for Patrick: your JD-MBA / Harvard Law Review background is uniquely positioned to evaluate this layer — that is why it has its own slide. Most VCs cannot read it.*

**Speaker script (3 min)**

This is the slide most VCs cannot evaluate, and I am putting it in the deck for you specifically.

The architecture this product requires is not just product-quality engineering. It is also the architecture EU AI Act, GDPR Article 17, China PIPL, and emerging AI liability frameworks are about to require for any product touching per-person behavioral data.

Five concrete examples — each pair is one regulation and the engineering already shipped against it. EU AI Act requires audit trail and provenance; we built immutable snapshot SSOT with publish-time fingerprints. GDPR Article 17 requires a real delete-me path with evidence; we have it. PIPL requires per-user scoped storage; we partition. Autonomous-self-modification liability is about to land; we built the gate. Per-persona refusal is required for any museum or licensed-IP contract; we ship it today.

The corollary that matters commercially: 50K enterprise channel through 1688 and Hengyi cannot legally sign without these. A corporate audit lawyer cannot put GPT-with-Memory on a regulated workflow; they can put Volvence on it. Governance is not a tax we pay — it is the moat that lets the enterprise channel open at all.

I am highlighting this slide because between you and me, this is the layer that gets undervalued in a Sand Hill Road pitch and correctly valued in a JD-MBA conversation.

**Design note**

The five-row table is the slide. Each row pairs regulation with shipped engineering. The footnote about Patrick's background should be small but not removed — it signals deliberate placement.

---

# Part B — Mobi as centerpiece

> The single largest signed JV. The thesis converted into one specific product with one specific partner and one specific number set. Three slides.

---

## Slide 7 — Mobi: the partner and the problem

**On-screen**

> **Mobi private-traffic JV — partner profile and structural problem**
>
> **Partner**: Mobi + 小小莎 IP resources + 乔栋's private-traffic pool. Combined audience **28 million followers**. Established MCN with proven private-traffic operations.
>
> **The structural problem they hire us to solve:**
>
> - Private-traffic conversion is bottlenecked by **human operator quality at scale**. A skilled 1-on-1 operator can convert 1-3% on a curated cohort. A scaled team of operators degrades quickly: average industry conversion ~0.3% (SCRM baseline).
> - The degradation is not skill; it is **memory and consistency**. Operator turnover, shift handoff, and broadcast templates produce a relationship that does not feel remembered.
> - Generic AI assistants do not solve it either — they have no per-user state, no preference scoping, no timing.
>
> **What Mobi pays for**: a digital employee that *behaves like the best 1-on-1 operator they have ever had, at scale of 28M*. Specifically:
>
> 1. Remembers each user's prior context across days and weeks
> 2. Never leaks one user's preferences into another's responses
> 3. Recommends only when relationship stage is right; holds back otherwise
> 4. Records typed feedback when user pushes back, and the next session reflects it
>
> **Status**: JV signed; pilot deployment in progress; first 90 days of conversion data not yet complete.

**Speaker script (2 min)**

Mobi is the JV we will walk through in detail because it is the most concrete commercial test of the relationship-continuity thesis.

The partner brings 28 million followers across a combined private-traffic pool. They have been running scaled human operators for years. They hit an industry-baseline conversion ceiling of about 0.3%, and the bottleneck is not operator skill — it is the operator's inability to remember each user consistently over weeks at scale.

That is exactly the problem the four runtime properties on Slide 5 are designed to solve. The next two slides show the conversation pattern that produces lift and the unit economics that translate lift into Volvence revenue.

**Design note**

Two-column layout. Left: partner profile and the SCRM 0.3% baseline. Right: the four properties (mirrored from Slide 5). The mirror is intentional — it shows the same four mechanisms now wearing partner-specific clothes.

---

## Slide 8 — Mobi: the conversation pattern (before / after)

**On-screen — paired conversations on the same user persona "Ms. Chen, new mother, joined private-traffic group via 小小莎 IP"**

> **Day 1** (both columns): Ms. Chen joins the group. Says she is a new mother. Asks about a single product.
>
> **Day 30 — SCRM baseline (~0.3% conversion):**
>
> ```
> Operator: "Hi! Spring promotion live — special bone broth bundle 30% off,
>           closes tomorrow!"
> Ms. Chen: [no response, opt-out within 2 weeks]
> ```
>
> - No memory of Day 1 context
> - Recommendation forced into a window the user is not ready for
> - Persona generic, voice indistinguishable from broadcast
>
> **Day 30 — Volvence relationship runtime (target 0.67% conversion):**
>
> ```
> Runtime (internal state, audit-visible):
>   user_state.lifecycle_stage = "newborn_4-6_weeks"  ← from Day 1 turn
>   user_state.no_bone_broth   = true                 ← from a Day 10 mention
>   relationship_state         = "early_trust_building"
>   recommend_now              = false                ← timing gate
>
> Runtime → Ms. Chen:
>   "Most parents around week 4-6 start noticing baby sleeping in
>    longer stretches but waking earlier. Is that happening for you?
>    Happy to share what other moms in the group have found useful
>    around now — only if you want."
> Ms. Chen: "Yes! She's sleeping longer but waking at 4am..."
>
> Day 45:
> Runtime: relationship_state has moved to "active_trust"; recommend_now = true
> Runtime → Ms. Chen: contextual product recommendation aligned with
>                     the 4am-wake conversation; product purchased
> ```
>
> **Four properties at work, each separately audit-logged:**
>
> 1. Cross-session memory — Day 1 lifecycle stage and Day 10 preference both retrieved on Day 30
> 2. Preference separation — no bone broth ever appears for Ms. Chen across all sessions
> 3. Recommendation timing — `recommend_now` gate held the system back at Day 30, opened at Day 45
> 4. Adaptation persistence — if Ms. Chen had said "stop pushing products," that becomes a typed event lasting across restart

**Speaker script (3 min)**

This is the actual pattern. Same Day 1; two different Day 30 outcomes.

SCRM baseline on the top: a broadcast push, no memory, no timing — opt-out within two weeks. This is the 0.3% industry conversion floor, and Mobi has hit that floor with skilled human operators because the operators cannot scale memory across thousands of users.

Volvence runtime on the bottom: Day 1 lifecycle context retrieved, a Day 10 dietary preference retrieved, a relationship-state gate holding the system back from recommending, a soft check-in that produces a real conversation, then the recommendation on Day 45 when the user has opened the topic herself. Audit-visible internal state on each line.

Four properties each separately audit-logged. Each one of them is one of the four LTV levers on Slide 5. Each one of them gets a cohort curve in the pilot.

I want to be clear: this conversation is the *runtime* in action; it is what the engineering produces. **What we still need to prove on Mobi is whether the lift translates to 0.5-1.0% conversion at scale.** That measurement starts now.

**Design note**

The slide should look almost like a diff. Two parallel conversation columns; on the Volvence side, the internal runtime state box renders in a slightly different shade (audit-trail color). The four-property footer is the bridge to Slide 9.

---

## Slide 9 — Mobi: unit economics, kill criterion, repricing thesis

**On-screen**

> **Mobi private-traffic JV — unit economics (figures direct from internal financial model)**
>
> | Item | Per converted user / year | 2026 target scale | 2026 Volvence revenue |
> |---|---|---|---|
> | Service fee (JV procures token from Volvence) | RMB 30 | 187,000 orders | RMB 2.8M |
> | JV profit share (50% allocation × 30% Volvence cut) | RMB 100 distributable | 187,000 orders | RMB 2.8M |
> | **Mobi JV 2026 subtotal** | | | **RMB 5.6M (~US$800K)** |
>
> **Three-year ramp on this single JV** (from internal model):
>
> | Year | Orders | Volvence revenue (RMB) | Volvence revenue (US$, ~7:1) |
> |---|---|---|---|
> | 2026 | 187,000 | 5.6M | ~US$0.8M |
> | 2027 | 561,000 (3× growth) | 16.8M | ~US$2.4M |
> | 2028 | 1,122,000 (6× from 2026) | 33.6M | ~US$4.8M |
>
> **Conversion assumption — internal model 0.67%, projected, not proven:**
>
> - SCRM industry baseline: ~0.3%
> - Volvence target with relationship runtime: 0.6-1.0%
> - 3-month pilot observation window not yet complete
> - **Kill criterion: 3-month pilot conversion < 0.5% → vertical deprioritized**
>
> **Repricing thesis** (why this is a different category from SCRM, not a cheaper SCRM):
>
> | Weimob / Youzan (SCRM baseline) | Volvence |
> |---|---|
> | Reach tools | Relationship engineering |
> | Broadcast / auto-reply | Remembered and understood |
> | One-time conversion | Cross-session LTV |
> | ~RMB 1K / month / brand | RMB 5K-50K / month / brand (target) |

**Speaker script (3 min)**

Two revenue lines per converted user per year on Mobi: RMB 30 service fee, RMB 100 distributable profit share. Volvence's take is 100% of the service fee plus 50% × 30% of the profit share. At 187K target orders in 2026, RMB 5.6M — roughly US$800K — from this single JV.

The internal three-year ramp is 1× → 3× → 6×. By 2028 this single JV produces RMB 33.6M, roughly US$4.8M, in Volvence revenue.

The conversion line is the only assumption you should pressure-test. SCRM industry baseline is 0.3%. Internal model is 0.67%. The kill criterion is 0.5% — below that we deprioritize the vertical, not double down. Three-month pilot observation window is in progress.

The larger commercial story is repricing. Weimob and Youzan sell reach at about RMB 1K per brand per month. Volvence targets RMB 5K to RMB 50K per brand per month for *relationship optimization*. The valuation multiple comes from being a different category, not a cheaper Weimob — and the runtime architecture is what makes that category claim defensible.

**Design note**

The three tables are the slide. `projected` and `kill criterion` are visually unmissable. Green border on the kill-criterion box. The three-year ramp table is critical for the LP read.

---

# Part C — Distribution, plan, technical credibility

---

## Slide 10 — Six signed JVs, four scenarios, one engine (full numbers from internal model)

**On-screen — all revenue figures from internal model**

> | Scenario | Partner | Audience / base | 2026 conversion / scale | 2026 Volvence revenue |
> |---|---|---|---|---|
> | **Companion** (UploadLive + influencer JVs) | 15M-follower influencer + others | 45M follower base | 1% activation × 10% conversion × US$42/yr × 30% share | ~US$1.87M |
> | **Parenting** (高盖伦 B2B2C: hardware + parenting app) | 11M-follower 高盖伦 | 60K units (0.55% conversion) × (RMB 500 hw + RMB 360/yr APP) | Internal model: RMB 8.4M | **~US$1.20M** |
> | **Private-traffic** (Mobi/小小莎/乔栋) | 28M-follower MCN | 187K orders (0.67% conversion) | Internal model: RMB 5.6M | **~US$0.80M** |
> | **Cross-border SaaS** (恒一 B2B2B / 1688) | 50K enterprises | 2,000 SaaS accounts × RMB 30K/yr | Internal model: RMB 6.0M | **~US$0.86M** |
> | **C-end self-operated** (Mira) | direct-to-consumer | 50K paid users × RMB 110/yr | Internal model: RMB 5.5M | **~US$0.79M** |
> | **Custom enterprise** (浙江数字国贸, 雷神等离子, 唐商文化) | signed contracts | RMB 6M + RMB 1M + RMB 2.5M | Internal model: RMB 9.5M | **~US$1.36M** |
> | **Total 2026 (internal model)** | | | | **RMB 35M ≈ US$5.0M** |
>
> **Six signed JVs**: 4 already signed, 2 signing in April (one US$200K deal already closed; 30K-enterprise partner signing).

**Speaker script (2 min)**

Same engine, six bundles, all figures from internal financial model.

Companion runs on the 45M follower base through UploadLive and influencer JVs.

Parenting is the JV with 高盖伦 — 11M followers, 60,000 hardware-plus-app units in 2026, RMB 8.4M.

Mobi private-traffic — the slide we just walked through — RMB 5.6M.

Cross-border SaaS through 恒一 / 1688 reaches 50,000 enterprises; 2,000 accounts × RMB 30K per year = RMB 6M.

C-end Mira self-operated is RMB 5.5M. Custom enterprise contracts — 浙江数字国贸 (signed), 雷神等离子 (in approval), 唐商文化 (signed) — add another RMB 9.5M.

Total internal model 2026: RMB 35M, roughly US$5M.

**Design note**

Full revenue map. The "Total 2026 ≈ US$5M" row in bold; everything above is the build. No projections beyond 2026 on this slide — those are the next slide.

---

## Slide 11 — 2026-2028 financials + 18-month proof plan

**On-screen**

> **2026-2028 ARR (RMB → US$ at ~7:1; figures from internal model)**
>
> | Year | Revenue | Net margin | Net profit |
> |---|---|---|---|
> | **2026** | RMB 35M (~US$5.0M) | **31.1%** | RMB 10.9M (~US$1.55M) |
> | 2027 | RMB 165M (~US$23.6M) | 46.2% | RMB 76.3M (~US$10.9M) |
> | 2028 | RMB 408M (~US$58M) | 54.2% | RMB 221M (~US$31.6M) |
>
> Conservative anchor: **2026 between US$3.33M and US$5.0M** depending on JV ramp speed. 2027-2028 are scenario, not proof.
>
> Cost structure (fixed vs. variable in 2026):
>
> - **Project delivery (variable with revenue): RMB 16.2M** (compute / API ~RMB 9.3M; delivery ~RMB 3.5M; ops ~RMB 0.7M; JV onsite staff ~RMB 2.7M)
> - **R&D (fixed): RMB 2.8M**
> - **Sales & marketing (fixed): RMB 3.3M**
> - **G&A (fixed): RMB 1.8M**
> - Total fixed overhead ~RMB 7.9M (~US$1.1M)
>
> Asset-light SaaS; we benefit when substrate inference costs decline.
>
> **18-month proof plan — signed access → recognized ARR**
>
> | Timeline | Milestone | Success criterion |
> |---|---|---|
> | M0-M3 | Mobi lighthouse in production | Real users; D30 retention curve produced; conversion data preliminary |
> | M4-M9 | 3 JVs in production | Repeatable deployment process; at least 2 of 4 LTV levers (Slide 5) show measurable lift |
> | M10-M18 | **Recognized ARR > US$1M real** | Not projected; recognized revenue + one full retention cohort |
>
> If we do not hit M9 or M18, we do not raise Series A on this story.

**Speaker script (3 min)**

Internal model is detailed; full file is available for DD.

Anchor on 2026 conservative band: US$3.3M to US$5M. The top of that band is internal model 2026. The bottom is a 33% haircut to reflect JV ramp risk.

Net margin trajectory 31% → 46% → 54% is driven by three things: variable cost (compute and delivery) scaling sub-linearly with revenue; fixed overhead diluting as engine scales; substrate cost falling as frontier labs lower inference price. Total fixed overhead in 2026 is roughly RMB 8M, US$1.1M.

The 18-month plan converts signed access into recognized ARR and into the four cohort curves on Slide 5. By M9: three JVs in production and at least two of the four LTV levers showing measurable lift. By M18: more than US$1M in recognized real ARR, not projected, plus one full retention cohort.

If we do not hit M9 or M18, we do not raise Series A on this story. That discipline is what makes this round investable.

**Design note**

The three-row financial table is the slide. Note explicitly that conservative anchor is the 2026 band; do not show 2028 as headline.

---

## Slide 12 — Technical credibility, in one frame

**Claim**

> Two things in one slide: (a) the *math* that says small-data continual learning works — twenty years of Yang Liu's work — and (b) the *engineering* that ships today. No arXiv IDs in the body; the deep paper anchors are in Appendix B / E.

**On-screen — left column: the math**

> **The hard question this product depends on:** real users only produce a few hundred turns of data per month. Can a system learn meaningful per-user adaptation from that?
>
> **Yang Liu's life work answers this question affirmatively.** *Minimax Analysis of Active Learning* (Hanneke & Yang, JMLR 2015) is the foundational result: under standard noise assumptions, **a learner that actively chooses which data to learn from can match a passive learner's accuracy with exponentially less data.**
>
> 40+ papers across active, online, transfer, and drifting-distribution learning — 18 at A-tier venues — provide the algorithmic backbone for the regime real consumer relationships actually produce: small data, non-stationary, continually arriving.
>
> The science was built before the company; the company sits on the science.

**On-screen — right column: the engineering**

> **What ships today, verifiable in repo:**
>
> | Asset | Number | Verify at |
> |---|---|---|
> | Contract tests passing in CI | **1063+ existing · 96 new** | Phase 1 exit-evidence doc |
> | Closed-alpha API serving real users | **Live**, with allowlist + scoped deletion + weekly report | `docs/closed-alpha-api-service.md` |
> | Public benchmark, Apache 2.0 | **Companion Bench v1.0** · 24 public + 96 held-out | `packages/companion-bench/` |
> | Per-user memory probes (pass where baseline RAG fails) | **4/4 PASS** (context, temporal, update, association); RAG fails 3/4 | `tests/longitudinal/test_vz_memprobe_*.py` |
> | OpenAI-compatible facade (read-only) | Any OpenAI SDK client connects with zero changes | `lifeform-openai-compat` |

**Speaker script (3 min)**

Two things on this slide and I want you to remember both.

On the math: small-data continual learning is not magic, it is a known theoretical regime. Yang Liu's foundational 2015 JMLR result proves that a learner that actively chooses which data to learn from matches a passive learner's accuracy with *exponentially* less data. Forty more papers refine the apparatus — full list in the appendix. That is why we can plausibly say "we will adapt to a user from a few hundred turns a month, not millions of episodes." Most teams in our space picked their algorithm in 2024; ours has a Chief Scientist whose PhD was in the math the product depends on.

On the engineering: five numbers verifiable in due diligence. 1063 existing contract tests plus 96 new, zero regression. Closed-alpha API live with real users. Companion Bench v1.0 open-sourced under Apache 2.0. Memory probes passing 4 of 4 where baseline retrieval-augmented generation fails 3 of 4. OpenAI-compatible facade so any existing SDK client connects with zero changes.

This is architecture as discipline, not as story.

**Design note**

Two equal columns. Left: "The math (Yang Liu, 20 years)." Right: "The engineering (verifiable in DD)." Keep arXiv IDs out; they are in Appendix B / E.

---

# Part D — Risks, ask, close

---

## Slide 13 — Risks (with falsifiers)

**On-screen**

> | Risk | What would prove / disprove |
> |---|---|
> | JV access does not convert to usage | Mobi lighthouse fails to retain users / produce conversion signal in 3 months |
> | Relationship continuity does not produce lift | Mobi 3-month conversion < 0.5% → deprioritize vertical; if 2 of 4 LTV levers fail by M6 → thesis softens |
> | Frontier labs ship deep memory + persona + state at production quality | We compete on **vertical depth, governance, JV distribution**, not on the memory primitive itself; thesis softens but commercial moat through signed JVs remains |
> | Regulation softens (EU AI Act / PIPL enforcement delayed) | Governance moat narrows; commercial moat through unit economics remains independent |
> | Team execution bandwidth | 3 JVs in production by M9 — otherwise we narrow focus |
> | Conversion lift exists but not at the magnitude needed for repricing thesis | Volvence positions as premium SCRM at RMB 3-5K / brand / month rather than RMB 5-50K — still a category, narrower upside |
>
> Each row is a falsifier, not a hedge. The round is for converting these risks into evidence.

**Speaker script (2 min)**

Risks I would focus on in your seat.

The first two are commercial — the lighthouse and the lift magnitude. The third is the most honest one: if a frontier lab ships deep memory + persona + state at production quality next year, we lose part of our framing — and our response is **not** "they cannot do it." Our response is that we compete on vertical depth, governance, and signed JV distribution. The commercial moat is real even if the primitive moat narrows.

The fourth is regulatory. If EU AI Act enforcement softens — possible — governance moat narrows; commercial moat remains.

The fifth is team — 3 JVs by M9 or we narrow.

The sixth is the most subtle. Even if relationship continuity does produce lift, it may not produce *enough* lift to support the RMB 5-50K / brand / month repricing thesis. In that case we are still a category — premium SCRM at RMB 3-5K — but a narrower upside. We will tell you which world we are in by M12.

**Design note**

Each row reads as falsifier; the third row in particular should be visually highlighted — that is the maturity tell.

---

## Slide 14 — The Ask

**On-screen**

> **Late Seed / Pre-Series A**
>
> | Dimension | Target |
> |---|---|
> | Round size | **US$3M - US$5M** |
> | Pre-money valuation (range, under discussion) | **US$20M - US$30M** |
> | Xfund target ticket | **US$1.5M - US$2.5M**, lead or co-lead |
> | Equity to Xfund | ~7% - 10% |
> | Runway | 18 months |
>
> **Use of funds**
>
> - **Engineering 40%** — runtime, deployment reliability, cohort-measurement instrumentation
> - **Compute / data 25%** — substrate, evidence pipeline, A/B cohort runs
> - **GTM 20%** — 3 in-production JV launches, lighthouse customers, partner success
> - **Operations / legal / IP 15%** — audit, consent, deletion, IP structure
>
> **Next financing gate**: 3 JVs in production, ARR > US$1M real, repeatable deployment playbook, at least 2 of 4 LTV levers (Slide 5) with measurable lift.

**Speaker script (2 min)**

US$3M to US$5M, late seed or pre-Series A, Xfund as lead or co-lead if there is alignment. Pre-money band US$20M to US$30M, under discussion. Xfund ticket roughly US$1.5M to US$2.5M, 7-10% equity at this band.

The round is for *measurement*, not expansion. Eighteen months to convert signed access into recognized ARR and to produce the four cohort curves on Slide 5. If we hit that, Series A from a position of evidence; if not, we narrow.

---

## Slide 15 — Close: the thesis in one frame

**On-screen**

> **The Volvence thesis, in five lines:**
>
> 1. Vertical proprietary data beats LLM scaling. (Xfund thesis, validated by Open Evidence.)
> 2. Generation 1 was institutional vertical data — Mayo Clinic × Open Evidence.
> 3. **Generation 2 is the per-person trajectory** — renewable, non-transferable, only captured by a runtime with memory, identity, audit, and consent.
> 4. **Relationship continuity creates economic lift** — four runtime properties → four LTV levers — and the next 12 months convert that hypothesis into cohort curves on Mobi and 2 other JVs.
> 5. Frontier labs optimize for generality across one billion users; we optimize for relationship state inside one vertical. Both can be real.
>
> Team: Hacker + Humanist; CMU-trained Chief Scientist whose math is the field.
> Engineering: 1063+ contract tests, closed-alpha live, governance shipped.
> Distribution: 45M followers, 50K enterprises, 6 signed JVs.
>
> Conservative 2026 ARR: **US$3.3M - US$5M**.

**Speaker script (60s)**

Five lines. Each backed by a citation, a number, or a JV.

The center bet is line 4: relationship continuity creates economic lift, measurable as four cohort curves, killable if any of them fail to move. Mobi is the first measurement; the next eighteen months are about producing the curves.

Happy to spend the rest on your questions.

---

# Optional Appendix / Q&A

> Use only when asked. Appendix E is the full v6/v7 technical thesis preserved for a senior technical DD partner.

---

## Appendix A — Yang Liu's academic record (full table)

**Importance scoring is internal (1 = highest). Venues cited are public.**

| Importance | Thesis | Venue | Date |
|---|---|---|---|
| 1 | *Bandit Learnability can be Undecidable* | COLT | Jul 2023 |
| 2 | *Active Learning with Identifiable Mixture Models* | In submission to Annals of Statistics | 2023 |
| 3 | *Reliable Active Apprenticeship Learning* | ALT | 2025 |
| 4 | *Toward a General Theory of Online Selective Sampling: Trading Off Mistakes and Queries* | AISTATS | Apr 2021 |
| 5 | *Computing and Testing Small Connectivity in Near-Linear Time and Queries via Fast Local Cut Algorithms* | SODA | Jan 2020 |
| 6 | *Statistical Learning under Nonstationary Mixing Processes* | AISTATS | Apr 2019 |
| 7 | *Surrogate Losses in Passive and Active Learning* | EJS | Nov 2019 |
| 8 | *A Theory of Transfer Learning with Applications to Active Learning* | Machine Learning | Feb 2013 |
| 9 | ***Minimax Analysis of Active Learning*** (key citation for Slide 12) | **JMLR** | **Jan 2015** |
| 10 | *Identifiability of Priors from Bounded Sample Sizes with Applications to Transfer Learning* | COLT | Jul 2011 |
| 11 | *Active Learning with a Drifting Distribution* | NeurIPS | Dec 2011 |
| 12 | *Learning with a Drifting Target Concept* | ALT | Oct 2015 |
| 13 | *Buy-in-Bulk Active Learning* | NeurIPS | Dec 2013 |
| 14 | *Active Property Testing* | FOCS | Oct 2012 |
| 15 | *Bounds on the Minimax Rate for Estimating a Prior over a VC Class from Independent Learning Tasks* | ALT | Oct 2015 |
| 16 | *Bayesian Active Learning Using Arbitrary Binary Valued Queries* | ALT | Oct 2010 |
| 17 | *Activized Learning with Uniform Classification Noise* | ICML | Jun 2013 |
| 18 | *Online Learning by Ellipsoid Method* | ICML | Jun 2009 |
| 19 | *Online Allocation and Pricing with Economies of Scale* | WINE | Dec 2015 |
| 20 | *Risk-Averse Matchings over Uncertain Graph Databases* | ECML PKDD | Sep 2018 |

**Plus 20+ more in TPAMI, NeurIPS, CVPR, AAAI, AISTATS, TCS, ITCS, UAI; full list available on request.**

---

## Appendix B — Citation index for the body deck

**Generation 2 vertical proprietary data:**
- Villalobos, Sevilla et al., *Will We Run Out of Data?* (Epoch AI, 2024) — internet text ceiling 2026-2032
- Open Evidence's Mayo-Clinic moat (Xfund portfolio, public)
- Delphi (Xfund portfolio, public)

**Bloomberg / Visa / Salesforce / TikTok analogy:**
- Public knowledge of each platform's interaction-graph moat. The analogue is structural, not metric.

**Yang Liu's small-data continual learning result:**
- Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015) — full apparatus in Appendix A

**Regulatory pressures:**
- EU AI Act (high-risk per-user systems, 2026-2027 enforcement)
- GDPR Article 17 (right to be forgotten)
- China PIPL (cross-border per-user data)
- Emerging AI liability frameworks (autonomous self-modification)

---

## Appendix C — Anti-claims (what we are *not* selling)

| Claim we are *not* making | Why we are not making it |
|---|---|
| "Smarter than GPT/Claude" | Substrate ceiling; IQ scaling is not our moat. We inherit substrate and amplify with relationship-state runtime. |
| "AGI in 12-24 months" | We do not say what we do not believe. Strong AGI probability < 5% in 24 months. |
| "Generic memory plugin" | OpenAI Memory / Mem0 / Letta own that lane. We do not compete in it. |
| "Agent framework" | LangChain / AutoGen own that. Our contract runtime is for our own use. |
| "AI psychologist / AI doctor" | Licensure / liability / regulation make this off-limits today. |
| "Companion for minors" | Legal and ethical risk too high for this stage. |
| "Unauthorized resurrection of living public figures" | Legal and ethical non-starter. |
| "OpenAI cannot do X" | Frontier labs optimize for generality across one billion users; we optimize for relationship state inside one vertical. Both can be real. |

---

## Appendix D — 60-second verbal version

> If Patrick says "tell me in one minute":

> *We are building the runtime that captures the next layer of vertical proprietary data — the relationship itself. Mayo Clinic data was layer one; Open Evidence is the proof. Layer two is the per-person trajectory each user produces with a product over months and years, renewable and non-transferable. The empirical question — and what this round is for — is whether sustaining that trajectory through persistent memory, preference scoping, recommendation timing, and adaptation after feedback measurably moves retention, conversion, and LTV. Four runtime properties become four LTV levers. We have the team — me as builder-operator, Yang Liu as CMU-trained Chief Scientist whose math on small-data continual learning is the actual field this product needs. We have the engineering — 1063+ contract tests, closed-alpha API live, governance architecture already shipped. And we have the distribution — 45M followers and 50K enterprises through six signed JVs. Mobi is the first measurement: 28M follower pool, internal model 0.67% conversion vs SCRM baseline 0.3%, 187K orders in 2026, RMB 5.6M Volvence revenue, growing to RMB 33.6M in 2028. Total internal model 2026 is RMB 35M, roughly US$5M, with 31% net margin. Frontier labs optimize for generality; we optimize for relationship state. Both can be real. Ask is US$3M-5M, Xfund as lead or co-lead, 18 months to convert signed access into recognized ARR and four cohort curves.*

---

## Appendix E — For technical DD partner: the architecture and the research thesis

> This appendix carries the full v6/v7 technical chain — Cognitive AGI framing, token-RL impossibility result, Nested Learning + Emergent Temporal Abstractions architecture, our four matched-control ablations, 28+ EQ benchmarks — for the senior technical DD partner who will take the conversation past the business-metric layer.
>
> **It is deliberately walled off from the body deck.** The body argues an empirical, falsifiable, business-metric thesis (relationship continuity → economic lift). This appendix argues the research thesis that sits underneath. The two are consistent; the second is not required to underwrite the first.

### E.1 — The 8-step research thesis

**Step 1.** Cognitive AGI, not world models or actuators, is the binding constraint on usable AI. Substrates (LLMs, world models, humanoids) are necessary; cognition exercised in real environments is what rides on top.
- *Sutton & Silver, The Era of Experience* (DeepMind, Apr 2024); Sutskever NeurIPS 2024 keynote.

**Step 2.** Cognitive AGI must be online continual learning on neural substrate. Hand-engineered prompts / harnesses / tool graphs are the new expert systems.
- *Sutton, The Bitter Lesson* (2019); 2025-2026 production failures of agent-harness companies.

**Step 3.** Humans, not the internet, are the next durable vertical data layer.
- *Will We Run Out of Data?* (Epoch AI, 2024); Open Evidence's Mayo-Clinic moat as Generation 1 precedent.

**Step 4.** Token-level RL is structurally infeasible — three labs proved this in 5 months.
- Anthropic, *Natural Emergent Misalignment from Reward Hacking* (Nov 2025)
- OpenAI + academia, *Reasoning Models Struggle to Control their Chains of Thought* (Mar 2026)
- MATS, *Output Supervision Can Obfuscate the CoT* (Nov 2025)
- Anthropic + Schulman, *Reasoning Models Don't Say What They Think* (2025)

**Step 5.** The path is emergent multi-timescale RL on a learned abstraction space `z_t` with switching gate `β_t`, trained with sparse interaction data.
- Behrouz & Mirrokni, *Nested Learning* (Google Research, arXiv:2512.24695)
- ETH-Sacramento, *Emergent Temporal Abstractions* (arXiv:2512.20605)
- Precup & Klissarov, *Discovering Temporal Structure: HRL Overview* (DeepMind, 2026)
- Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015)

**Step 6.** Our three answers.
- **Reward = Body.** Botvinick / Wang / Dabney 2025 *Distributional Dopamine* + Friston *Active Inference*. Implementation: `vz-cognition.prediction` + `vz-cognition.credit` modules with drive states.
- **Abstractions emerge from Nested Learning + ETA.** `CMSVariant.NESTED` shows initialization error decreasing monotonically across context resets. `scripts/run_eta_paper_suite.sh` runs 4 matched-control ablations — `full-no-optimize`, `full-no-replacement`, `learned-lite-causal`, `noop-backend` — all 4 PASS on hierarchical sparse-reward, abstract-action family reuse, held-out composition, and delayed credit alignment.
- **Sparse data = Active Learning.** Hanneke & Yang JMLR 2015 + Yang's 40+ papers. VZ-MemProbe 4 probes PASS at production-relevant data scales where baseline RAG fails on 3 of 4.

**Step 7.** Thin Prompt, Thick Runtime — implemented and shipping.
- 96 new contract tests PASS; 1063+ existing zero regression
- 5 vertical lifeforms co-loaded in one process (CI-enforced)
- Closed-alpha API serving real users
- Companion Bench v1.0 open-sourced under Apache 2.0
- Figure vertical full chain reproducible byte-equivalent
- Rupture/Repair typed enum loop with durable memory and audit trail
- GDPR/PIPL deletion path with evidence ledger
- OpenAI-compatible facade

**Step 8.** Frontier labs are sprinting toward IQ scaling; relationship-state runtime is an orthogonal moat. They optimize for generality across one billion users; we optimize for vertical relationship state. Public market signal: GPT-5 is engineering integration; SSI silent; Karpathy → Eureka Labs; Schulman → Thinking Machines.

### E.2 — IQ and EQ emergence (28+ PASSing benchmarks)

**IQ** = Substrate × ETA abstract-action reuse × Nested-Learning persistent memory accumulation.

**EQ — four mechanisms × four academic anchors × 28+ independent PASSing tests:**

| Mechanism | Academic anchor | Repo benchmarks |
|---|---|---|
| Dual-track learning (`world_temporal` vs. `self_temporal`) | Premack & Woodruff 1978; ETA latent-track separation | `test_multi_party_scenarios.py` (10 PASS); cross-user isolation PASS |
| 4 Theory-of-Mind owners keyed by interlocutor | Saxe / Wellman developmental psychology | `test_feeling_about_other_active_matched_control.py` (8 PASS); common-ground matched control PASS |
| Rupture/Repair typed enum loop | Closed-loop emotional learning beyond thumbs-up/down | Three cross-session-hydration tests PASS |
| Regime persistent identity | DeepMind 2025-2026 work on training-time character | `test_affordance_delayed_credit.py` (4 PASS); composite dispatch ACTIVE |

**Total: 28+ independent PASSing benchmarks across `tests/contracts/`, `tests/longitudinal/`, `tests/test_social_*`.**

### E.3 — Why this appendix is not in the body

Because the empirical thesis (Slides 3-5) is independently underwritable on business metrics — the Mobi cohort curves either move or they do not — and because civilization-level claims compete for credibility budget that the company's stage cannot yet underwrite. The architecture is real; the business metrics are not yet proven. We argue the business case in the body, the architecture in the appendix, and let the seriousness of the architecture speak through the empirical lift it produces. That is the order in which a 25-year venture career would prefer to see them.

---

## Appendix F — Q&A

### Q1 — You have 6 JVs but no recognized ARR. Why is this traction?

Signed distribution access, not recognized revenue — and we say so on Slide 11. The 18-month plan converts at least three JVs into in-production deployments with recognized revenue plus the four cohort curves on Slide 5.

### Q2 — Why not just use GPT with memory once it ships?

GPT memory is a substrate. We will use it. Real human-relationship products require typed feedback enums, scoped deletion with audit, persistent runtime identity, rupture-repair loops, and per-persona refusal as commercial-contract requirements — not as architectural superiority claims. **Frontier labs optimize for generality across one billion users; we optimize for relationship state inside one vertical.** If GPT memory strengthens, our substrate improves; the runtime + governance + JV-distribution layer remains ours.

### Q3 — What is your unfair advantage?

Three things stacked. Scientific depth in non-stationary small-data learning through Yang Liu. Engineering discipline — 1063+ contract tests, closed-alpha live. Distribution — 45M followers, 50K enterprises through 6 signed JVs. None alone is enough; the combination is.

### Q4 — What would make you change direction?

Mobi 3-month pilot < 0.5% conversion → deprioritize Mobi. M6 without 2 of 4 LTV levers showing measurable lift → thesis softens. M9 without 3 JVs in production → narrow focus. Frontier labs ship deep memory + persona + state at production quality → we compete on vertical depth + governance + distribution rather than primitive.

### Q5 — What is the upside, not the SaaS upside?

Repricing in the near term — Weimob and Youzan sell reach at RMB 1K / brand / month; Volvence targets RMB 5K-50K / brand / month if conversion holds. Platform option in the long term — every brand running on Volvence accumulates a non-transferable per-user trajectory; that is the Generation-2 data asset.

### Q6 — Why Xfund specifically?

Three reasons. First, the thesis is a direct extension of Xfund's existing framing — proprietary vertical data beats LLM scaling — into the next layer. Second, Patrick underwrites founder judgment and category formation, which is what this round is. Third, your Delphi and Open Evidence portfolio are complementary slices: Delphi captures one person statically; Open Evidence captures institutional vertical data; Volvence captures the live trajectory with governance.

### Q7 — Your conversion rate of 0.67% is double the SCRM baseline. How confident?

This is internal-model projection, not measurement. The kill criterion below 0.5% is the discipline. We expect a noisy first 30 days, then a clearer curve by day 90. The runtime properties on Slide 5 — memory, preference scoping, timing, adaptation — each one should move conversion by a measurable amount; we will report which of them moves how much.

### Q8 — Why is governance a moat and not a tax?

Because the architecture required for a long-relationship product (per-user state, audit trail, scoped deletion, persistent identity) is the *same* architecture required by EU AI Act, GDPR Article 17, China PIPL, and emerging AI liability frameworks. We did not build governance for compliance; we built it because the product needs it, and regulation happens to point at the same thing. Our 50K-enterprise channel through 1688 cannot sign a workflow contract on GPT-with-Memory; they can on Volvence.

### Q9 — The technical depth in the appendix — is that real?

Yes. Appendix E carries the architecture chain — Cognitive AGI, latent-space RL, Nested Learning + Emergent Temporal Abstractions, four matched-control ablations PASSing, 28+ EQ benchmarks — with full paper citations, arXiv IDs, and script paths in repo. Any senior technical DD partner can reproduce. We deliberately keep that material out of the body because the round is being underwritten on business metrics, not on research claims.

---

# PPT Production Notes

## Visual style

- Black or near-black background.
- One main thought per slide.
- Dense tables only on Slides 5, 6, 9, 10, 11, 13.
- Use green only for emphasis (numbers, kill criterion, honest-disclosure box) — not decoration.
- On Slide 5, the **honest disclosure box is bordered in green** and the most prominent element on the slide.
- On Slide 8, the conversation diff is the visual lede — two columns, audit-trail color on the right column's internal state.
- On Slide 9, `projected` and `kill criterion` are visually unmissable.
- On Slide 3, the three-generation diagram is the visual lede of the entire deck. Spend layout time on it.
- No hype words on-screen.

## Demo handling

- Mobi demo is 4-5 minutes max.
- Subtitles English if conversation is Chinese.
- Highlight only 4 moments mapped to Slide 5's four LTV levers:
  - prior context remembered (retention)
  - preference separation (trust premium)
  - recommendation timing (conversion)
  - adaptation after feedback that persists across restart (recovery)

## Speaker behavior

- After naming a kill criterion or an honest-disclosure number — pause. Let the discipline register.
- If Patrick interrupts, switch to conversation. Slides 1-3, 5, 8-11 are the must-land set; everything else is supporting evidence.
- **Goal**: leave the room with Patrick able to repeat one sentence to his LP — *"Volvence is Generation 2 vertical proprietary data — relationship trajectory — and they have signed access to measure whether continuity creates economic lift within 12 months."*

## Words to avoid (body only — allowed in Appendix E)

- "AGI", "Cognitive AGI", "the era of experience", "Bitter Lesson"
- "frontier labs structurally cannot", "OpenAI cannot do X"
- "token-RL is dead", "alignment-faking"
- "Nested Learning", "Emergent Temporal Abstractions", "ETA", "β_t", "z_t", "metacontroller", "R1-R15"
- "唯一", "永远", "结构性独占", "杀伤力", "灵魂级", "打爆"

## Replacement language

- "Frontier labs optimize for generality; we optimize for relationship state."
- "Generation 2 vertical proprietary data."
- "Relationship continuity creates economic lift — we will measure whether it does."
- "Our current judgment is..."
- "This has not yet been proven; the kill criterion is..."
- "DD can re-run this live..."

---

## Change log

- **2026-05-17 v8.0**: Reorganized around empirical business-metric spine ("Relationship Continuity Creates Economic Lift"). Civilization-level narrative (AGI / cognitive AGI / token-RL impossibility / frontier-labs-cannot) removed from body and walled off in Appendix E for senior technical DD partner. New Slide 5 ("relationship continuity → 4 LTV levers") becomes the spine. Mobi promoted to centerpiece across Slides 7-9 with full unit economics drawn from internal financial model (`docs/business/00：19-详细版本(1).xls`): RMB 5.6M / 16.8M / 33.6M three-year ramp, 0.67% conversion vs SCRM 0.3% baseline, kill criterion < 0.5%. Slide 10 lists all six signed JV revenue lines from internal model. Slide 11 adds full cost structure breakdown and 3-year net-margin trajectory (31% → 46% → 54%). "Frontier labs structurally cannot ship X" language replaced everywhere with "they optimize for generality; we optimize for relationship state." Honest-disclosure box added to Slide 5 (no 6-month retention curve yet). New thesis-level kill criterion: 2 of 4 LTV levers must show measurable lift by M6.
- **2026-05-17 v7.0**: Reorganized for Patrick-Chung-specific profile (founder + thesis-extension + governance). Compressed v6's 8-step technical thesis from body to appendix.
- **2026-05-17 v6.0**: First draft of 8-step thesis chain with explicit Proof blocks.
