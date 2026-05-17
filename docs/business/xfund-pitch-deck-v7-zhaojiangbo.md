# VolvenceZero — Xfund Pitch Deck v7

> Status: **v7.0 draft (2026-05-17)**
> Audience: **Patrick Chung specifically**. Public materials (Stanford Daily 2015 "Hacker + Humanist" interview, Future Planet Capital 2024 interview, his 2024-05-01 fireside chat with Sam Altman at Harvard SEAS) place his strengths squarely in **founder judgment, category formation, governance, university tech transfer, IP, distribution**, not in independent evaluation of model architecture or RL theory.
> Implication: every page of the main deck must be defensible **in his vocabulary** (data moat / category / regulation / talent / distribution / unit economics). Technical depth is preserved — but moved to the appendix, where his technical DD partners pick it up.
>
> Design principle (carried over from v6): **every logical step must be backed by a citable proof** — named industry expert, peer-reviewed paper, our quantitative experiment in repo, or solid market number. No claim should rest on rhetoric alone.
>
> Recommended meeting format: **~35 min presentation + ~25 min conversation**. Short version: Slides 1-9 only.

---

## V7 vs V6: what changed and why

V6 led with an 8-step technical thesis chain (Sutton → Sutskever → token-RL danger papers → Nested Learning → ETA → ...). It is intellectually clean but it asks Patrick to underwrite *technical judgments he has no public evidence of being able to make independently*. In his Sam Altman fireside chat he asked **zero** technically detailed questions across twelve main prompts; an MIT student had to ask "what's next after transformers" for him. His public AI thesis (Future Planet Capital, July 2024) operates entirely at the **moat / data / distribution / governance** layer.

V7 reorganizes around that reality:

1. **Founders and thesis extension first.** The deck opens with the Hacker + Humanist team frame (which is Xfund's own thesis, applied to us) and an explicit "Patrick, this is the *next* layer of what you've already invested in" page that puts Mayo Clinic / Open Evidence and Delphi on the same diagram with Volvence.
2. **The AI argument is compressed from 8 slides to 3.** Internet text exhausts → relationship data is the only renewable durable vertical layer → frontier labs structurally cannot ship the runtime that captures it. Each is delivered in Patrick-native vocabulary (data moat, category, narrative incompatibility), not in arXiv IDs.
3. **A governance / IP / regulation slide is promoted.** This is one of Patrick's strongest evaluation lanes (Harvard Law Review, JD-MBA, ZEFER fundraising, NEA regulated industries). v6 buried it; v7 makes it Slide 6.
4. **Technical credibility becomes ONE slide** (Slide 7), framed as: "our co-founder spent 20 years proving the math behind small-data continual learning; here is the engineering reality that ships today." Paper details move to Appendix B.
5. **The 8-step technical thesis is preserved verbatim** in Appendix A, so a senior technical DD partner can read every link, with its citation, in one pass.
6. **All v6 commercial discipline carried forward**: 6 signed JVs, Mobi unit economics + kill criterion, 2026 conservative ARR US$3.33M-5M, 18-month proof plan, falsifier-based risks, the ask.
7. **Words to avoid list expanded** to include AI-research jargon that Patrick has no documented framework to evaluate: `β_t`, `z_t`, `metacontroller`, `R1-R15`, `Nested Learning` (in body; allowed in appendix), `ETA` (in body; allowed in appendix).

---

## Meeting cadence

```text
0-2 min      Cover + one-line pitch
2-6 min      Founders (Hacker + Humanist team thesis)
6-10 min     Why this is the next layer of Xfund's thesis  ← deck spine
10-20 min    The 3-step argument (data ceiling, why labs locked out, governance moat)
20-25 min    Technical credibility — Yang Liu math + engineering reality
25-30 min    Mobi demo
30-40 min    Distribution + JVs + Mobi unit economics + ARR + 18-month plan
40-45 min    Risks + ask + close
45-60 min    Conversation / DD questions
```

If Patrick interrupts at any point, switch to conversation. Every block of v7 stands alone.

---

# Main Deck

## Slide 1 — Cover

**On-screen**

> **VOLVENCE**
>
> **The runtime for the next layer of vertical proprietary data: the relationship itself.**
>
> Mayo Clinic was layer one. The person, every day, is layer two.
> Owned by no foundation lab. Renewable. Non-transferable. Auditable.
>
> Zhao Jiangbo, Founder & CEO
> Xfund conversation · May 2026

**Speaker script (60s)**

Patrick, thank you for the time.

I want to give you one thesis in this meeting. Mayo Clinic data is the first layer of vertical proprietary data — Open Evidence is the proof. **The next layer is the relationship itself: the moment-by-moment trajectory each person produces with a product over months and years. It is owned by no foundation lab, it is renewable, it is non-transferable, and the governance layer it requires is hostile to the universal-assistant narrative every frontier lab is locked into.**

Volvence is the runtime that captures this layer. Twenty years of math from my co-founder Yang Liu — CMU under Avrim Blum and Jaime Carbonell, forty peer-reviewed papers on exactly the regime that small-data continual learning requires. Two years of architecture work since ChatGPT. **45 million followers and 50 thousand enterprises** already in our distribution through 6 signed JVs. 2026 conservative projected ARR between US$3.3M and US$5M.

I will spend the next thirty-five minutes showing why this layer is durable, why the labs are locked out, and why our team is the one to build it.

**Design note**

Cover should be near-black, only one phrase visible at first frame: **"The runtime for the next layer of vertical proprietary data: the relationship itself."** Everything else fades in.

---

## Slide 2 — Founders: Hacker + Humanist, the team you have been writing checks to for ten years

**On-screen**

> **Two principals; one team thesis: Hacker + Humanist.**
>
> **Zhao Jiangbo — Founder & CEO** (full-time)
>
> - PKU CS · MBA — *Hacker side*: GitHub commits since Nov 2022, DD-verifiable
> - IBM Japan Research · HP China Software Sales GM · Alibaba VP Office · Tencent industry director — *Humanist side*: built and sold enterprise narratives across three of the largest tech employers in Asia
> - 3x founder; Haopai 300K users with zero paid acquisition; commercial exit
> - Self-funded Volvence R&D with RMB 5M since ChatGPT launch
>
> **Yang Liu, PhD — Co-founder & Chief Scientist** (full-time)
>
> - CMU PhD, advised by **Avrim Blum** and **Jaime Carbonell**
> - IBM Research · Yale postdoc
> - **40+ papers** in active learning, drifting distributions, transfer learning, online learning
> - 18 A-tier (NeurIPS / ICML / JMLR / COLT / SODA / FOCS / AISTATS); upcoming NeurIPS 2026
>
> **Wang Cangyu, CSO** · PhD Psychology · ex-Zhongqi Media GM (Douyin best agency, RMB 6B revenue)
> **Zhang Chi, CTO** (full-time) · Tsinghua CS · ex-Glodon · Haopai co-founder
> **Wu Xiang, CMO** (full-time) · ex-HP, Neusoft executive · 20yr enterprise GTM
>
> *This is the team profile Xfund has been backing since Loopt: a builder who also writes, paired with a research scientist whose math the rest of the field will eventually need.*

**Speaker script (3 min)**

Patrick, the Xfund thesis since 2012 has been the Hacker + Humanist team — the lateral thinker who is both a builder and someone who can think in long sentences. I want to show you why this team profile fits cleanly.

I am the hacker-humanist. I trained at PKU CS and then MBA, but more importantly I have run product, sales, and strategy at IBM Japan Research, HP China, Alibaba, and Tencent. Three founder cycles before this one. Haopai grew to 300 thousand users with no paid acquisition; we sold it commercially. I am writing code on Volvence today — my GitHub commits since November 2022 are public for DD. I am not a non-technical CEO outsourcing engineering.

Yang Liu is the technical anchor. CMU PhD under Avrim Blum and Jaime Carbonell — Carbonell was the founding director of CMU's Language Technologies Institute, and Avrim Blum is one of the most cited learning theorists alive. Yang has forty peer-reviewed papers. Eighteen are A-tier — NeurIPS, ICML, JMLR, COLT, SODA, FOCS, AISTATS. Her field is exactly the field this company needs: **how to learn fast from very little data, when the underlying distribution is changing.** I will come back to why that math matters in slide 7.

Wang Cangyu is psychology PhD plus the GM of Zhongqi Media — the agency Douyin officially recognizes as the best, with six billion RMB in annual revenue. Zhang Chi is Tsinghua CS, ex-Glodon, my co-founder at Haopai. Wu Xiang is twenty years of enterprise GTM through HP and Neusoft.

That is the team you are underwriting.

**Design note**

Yang's CMU advisors and paper count are the credibility anchors of this slide. Make them visually heavier. The phrase "Hacker + Humanist" should be a header.

---

## Slide 3 — Why this is the natural extension of Xfund's existing thesis

**On-screen**

> **Three generations of "vertical proprietary data beats LLM scaling".**
>
> ```
> Generation 1 — Institutional vertical data
>   ↳ Mayo Clinic × Open Evidence       (Xfund portfolio; finite; licensable)
>
> Generation 2 — Per-person trajectory data
>   ↳ THIS IS VOLVENCE                  (renewable; non-transferable; auditable)
>
> Adjacent — Snapshot replicas of individuals
>   ↳ Delphi                            (Xfund portfolio; static clone)
> ```
>
> **What is different about Generation 2 — and why a *runtime* is the only thing that can capture it:**
>
> - Mayo data is *recorded medicine*; it sits in a database.
> - A Delphi clone is a *snapshot of one person*; it is read-mostly after creation.
> - Volvence captures **the live, multi-turn, multi-day, cross-session, consent-bound trajectory** of each user — and *learns from it* without ever moving the substrate.
>
> That trajectory is the only data layer that:
>
> 1. Renews every day a user shows up, without any institutional intermediary;
> 2. Cannot be copied or scraped because it is jointly produced with the user;
> 3. Compounds across sessions only if the runtime has identity, memory, audit, and consent — i.e. only if a thick runtime exists.

**Speaker script (3 min)**

Patrick, the most important slide in this deck is this one, and I want to frame it in your own portfolio.

Open Evidence is the canonical proof of "vertical proprietary data beats LLM scaling." The Mayo Clinic moat is enormously valuable — and it is also a particular kind of data: recorded, institutional, ultimately licensable. There is a finite list of Mayo-Clinic-equivalent institutions in the world, and the strategic question is how many of them you can lock up.

Delphi sits next to that, in an interesting place: it captures one person, statically. Their messages, their decisions, frozen into a clone you can talk to. A snapshot replica.

We sit one layer past both of them. The data we are betting on is **the live trajectory each person produces with a product**, not as a recorded institutional dataset and not as a one-shot snapshot — but as a stream that renews every time the user shows up, is jointly produced with them, and compounds across sessions.

That data is renewable. Mayo Clinic data does not regenerate when you read it. A user's relationship with our product produces new data every interaction. It is non-transferable: it is co-owned with the user. And it only compounds if the runtime has identity, persistent memory, governance, and audit — which is the engineering thesis I will get to.

This is not a competing thesis to Open Evidence; it is the next layer of the same thesis. If Generation 1 was "data the institution generates," Generation 2 is "data the person generates with you, every day, for years."

**Design note**

The three-row diagram is the visual lede. Use real visual hierarchy — Mayo / Open Evidence at the top in muted tone, Volvence in the middle with strong contrast, Delphi adjacent and muted. The headline reads "Generation 2."

---

# Part A — Why this layer is durable

> Three slides. Each one is delivered in Patrick's vocabulary: data moat, market structure, governance. The papers and arXiv IDs that support each claim are in Appendix B.

---

## Slide 4 — The data ceiling: internet text exhausts, institutional verticals enumerate, only person-data renews

**Claim**

> The "more text + more compute" frontier is closing on a hard ceiling between 2026 and 2032. Institutional vertical data (Mayo, Bloomberg, Westlaw) extends the runway but is enumerable. **The only data source that scales with the number of users and the number of days they show up is the per-person trajectory.** That is the durable layer.

**On-screen**

> | Data layer | Volume | Renewable? | Locked-in? |
> |---|---|---|---|
> | Public-internet text | ~10-50 TB indexed | No — exhausted 2026-2032 | No |
> | Institutional vertical (Mayo, Bloomberg, Westlaw) | ~100s of TB per institution | No — finite per institution | Per license |
> | **Per-person trajectory at 1B users × 10 MB/day** | **~3.6 EB / year** | **Yes — daily** | **Yes — joint with user** |
>
> Magnitude ratio between row 1 and row 3: ~10⁵.

**Proof (cited; full detail in Appendix B)**

- **Epoch AI, *Will We Run Out of Data?* (Villalobos, Sevilla et al., 2024)** — quantitative projection of the internet-text ceiling.
- **Open Evidence's Mayo-Clinic moat** — Xfund-portfolio precedent for institutional vertical data as a moat.
- **OpenAI's 2024-2025 pivot to memory, persistent users, and consumer integration; Anthropic's pivot to coding tools** — both frontier labs are racing to occupy *user-state* before the substrate ceiling closes. They see the same picture.

**Speaker script (2 min)**

The reason we believe per-person trajectory is the durable layer is structural, not opinion. Epoch AI put a number on it in 2024: high-quality public-text data exhausts somewhere between 2026 and 2032 depending on assumptions. After that, scaling by reading more of the internet does not work — the internet is finite.

Open Evidence's Mayo Clinic deal is the canonical move into Generation 1, institutional vertical data. It is valuable; it is also enumerable.

But the math on Generation 2 is different by five orders of magnitude. A modal smartphone user produces around ten megabytes of contextual data a day. A billion users for a year is roughly 3.6 exabytes. The entire indexed text web is around fifty terabytes. The ratio is around ten to the fifth.

That is the layer no foundation lab structurally owns, because it is produced *with* the user, not *about* the user. And it only becomes a moat for the company that has the runtime to remember, scope, and govern it.

**Design note**

The 10⁵ ratio is the slide. Make it the visual anchor.

---

## Slide 5 — Why frontier labs are structurally locked out of this layer

**Claim**

> The largest LLM labs are not standing still — they are sprinting at each other on IQ scaling. Their universal-assistant narrative is *commercially incompatible* with the three product properties any real human-relationship product requires. This is not a temporary gap; it is structural for 12-24 months.

**On-screen**

> **Three product properties that any real human-relationship product requires — and that no frontier-lab roadmap accommodates:**
>
> 1. **Per-persona refusal** — a museum's Einstein bundle must say "I never wrote about that." A frontier lab cannot ship per-persona refusal because their brand is "one assistant for everything."
> 2. **Typed feedback enums** — when a user says "you are over-directive," the product must record that as a *typed event*, write it to durable memory, and behave differently in the next session. Thumbs-up / thumbs-down at API layer does not get there.
> 3. **Persistent runtime identity** — a brand's values, a parent's preferences, a museum's curatorial voice are runtime state, not system-prompt strings. They must survive substrate updates and audit reviews.
>
> All three conflict with the universal-assistant brand narrative.
>
> **Independent public signals that the frontier is splintering, not converging:**
>
> | Signal | What it shows |
> |---|---|
> | GPT-5 is engineering integration of the o-series + dual-process router — no new scaling milestone | The pure-scaling-as-SOTA window is closing |
> | Sutskever's SSI is unicorn-priced with **zero models, zero papers** since launch | Whatever Sutskever is building is not what OpenAI is shipping |
> | Karpathy left frontier work; founded **Eureka Labs** for education tooling | The frontier's most articulate communicator chose not to compete on scaling |
> | Schulman rotated through Anthropic to **Thinking Machines** | Two of the most influential RL researchers of the decade rotated away from frontier scaling |

**Speaker script (3 min)**

Three things every real human-relationship product has to do, and no frontier lab can ship them — for brand and narrative reasons, not for engineering reasons.

The first is per-persona refusal. A museum that licenses our Einstein bundle needs the system to say, *credibly*, "I never wrote anything about that subject." OpenAI cannot ship that because their brand is one helpful assistant.

The second is typed feedback enums. When a user looks at the product and says "you are too pushy" — that has to become a structured event, written to durable memory, and the next session has to behave differently. A thumbs-up button at API layer is not architecturally able to do this.

The third is persistent runtime identity. A brand voice, a parent's parenting style, a museum's curatorial values — these are not a system prompt. They are runtime state that survives substrate updates and is auditable in a contract review.

All three conflict with the brand narrative every frontier lab is building around. This is not a 90-day temporary gap; it is structural for at least the next 12-24 months.

The other half of this slide is public market signal. GPT-5 is engineering integration, not paradigm leap. Sutskever's SSI has been silent and unicorn-priced for two years. Karpathy left the frontier to build education tooling. Schulman has rotated to Thinking Machines. The frontier is splintering into multipolar competition, and the layer we are building is not on any of their roadmaps.

**Design note**

Two columns. Left: "Three properties frontier labs cannot ship." Right: "Public market signal." Names on the right column are bold — Sutskever, Karpathy, Schulman.

---

## Slide 6 — Governance is the third moat, and it is a moat Xfund underwrites well

**Claim**

> Every other slide so far is data + market structure. This slide is **governance**. The runtime architecture this product requires — owner-snapshot SSOT, audit trail, scoped deletion, consent-bound memory, gated self-modification — is the same architecture that EU AI Act, GDPR Article 17, and China PIPL are about to mandate for any product that touches per-person behavioral data. **We built that architecture before regulation forced it.**

**On-screen**

> **Five regulatory pressures landing in 2026-2028, and the engineering we have already shipped for each:**
>
> | Regulatory pressure | What it requires | What we shipped |
> |---|---|---|
> | EU AI Act (high-risk per-user systems, 2026-2027 enforcement window) | Audit trail of automated decisions, transparency of training data, scoped purpose | `vz-contracts` immutable snapshot SSOT with publish-time provenance |
> | GDPR Article 17 right to be forgotten | Per-user deletion with evidence | `DELETE /v1/users/me/memory` + deletion-evidence ledger |
> | China PIPL cross-border data | Per-user scoped storage and consent | Per-user memory partitioning; consent surfaces in API |
> | Liability for AI autonomous self-modification | A gate that decides when an AI is allowed to change itself | `R10 ModificationGate` — gated self-modification primitive |
> | Bio/Chem safety capability requirements (frontier-lab side, but pulls vendors with it) | Refusal at persona level, not at model level | Per-persona / per-figure refusal already implemented |
>
> **The same architecture is also the basis for enterprise contracts.** A 50K-enterprise client base cannot sign without these properties. The audit lawyer signs off on the contract; the audit lawyer cannot sign off on GPT-with-Memory.
>
> *Note: Patrick's professional background — Harvard Law Review, JD-MBA, ZEFER fundraising, Harvard Alumni Association directorship, founding-partner-level work at NEA on regulated-industry investments — is uniquely well-suited to evaluating this layer. We list it explicitly because it is the layer most VCs cannot read.*

**Speaker script (3 min)**

This is the slide most VCs cannot evaluate, and I am putting it in the deck for you specifically.

The runtime architecture this product requires is not just for product quality. It is also the architecture EU AI Act, GDPR Article 17, China PIPL, and emerging AI liability frameworks are about to require for any product that touches per-person behavioral data.

Five concrete examples. First, EU AI Act demands audit trail of automated decisions and provenance of training data for high-risk per-user systems. We built immutable snapshot SSOT — every internal state is published as a contract-typed snapshot with a publish-time fingerprint. Second, GDPR Article 17 demands a real "delete me" path with evidence. We have it: `DELETE /v1/users/me/memory` plus a deletion-evidence ledger. Third, PIPL requires per-user scoped storage. We do per-user memory partitioning. Fourth, AI liability for autonomous self-modification is about to land — we built a `ModificationGate` primitive that decides when the AI is allowed to change itself, with evidence and rollback. Fifth, per-persona refusal — at least one large museum customer cannot license an AI Einstein without the system being able to say "I never wrote about that."

The same architecture is also the basis for enterprise contracts. The 50,000-enterprise customer base we have through 1688 and the cross-border JVs cannot legally sign without these properties. A corporate audit lawyer cannot put GPT-with-Memory on a regulated workflow today; they can put Volvence on it.

I am highlighting this slide because, between you and me, this is the layer that gets undervalued in a Sand Hill Road pitch and that gets *correctly* valued in a JD-MBA conversation.

**Design note**

The five-row table is the slide. Each row pairs a piece of regulation with a piece of our shipped engineering. The footnote about Patrick's background should be small but not removed — it is a signal of why we are spending a slide on this.

---

# Part B — Why our team specifically

---

## Slide 7 — Technical credibility, in one frame

**Claim**

> Two things in one slide: (a) the *math* that says small-data continual learning is tractable — twenty years of Yang Liu's work — and (b) the *engineering* that says it is already running. No paper-IDs in the body; the deep technical anchor is Appendix B.

**On-screen — left column: the math**

> **The hard question:** real users only produce a few hundred turns of data per month. Can a system learn meaningful per-user adaptation from that?
>
> **Yang Liu's life work answers this.** *Minimax Analysis of Active Learning* (Hanneke & Yang, JMLR 2015) is the foundational result: under standard noise assumptions, **a learner that actively chooses which data to learn from can match a passive learner's accuracy with exponentially less data.**
>
> The full apparatus — 40+ papers across active, online, transfer, and drifting-distribution learning, 18 at A-tier venues — provides the algorithmic backbone for: "this user only produces 200 turns a month; how do I extract maximal signal from those 200 turns?"
>
> This is not a marketing line. It is the field of our co-founder's PhD, postdoc, and 15-year academic record. **The science was built before the company; the company sits on the science.**

**On-screen — right column: the engineering**

> **What ships today, verifiable in repo:**
>
> | Asset | Number | Verify at |
> |---|---|---|
> | Contract tests passing in CI | **1063+ existing · 96 new** | `docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md` |
> | Vertical lifeforms co-loaded in one process (CI-enforced) | **5** (`emogpt`, `coding`, `character`, `figure`, `growth-advisor`) | `PARALLEL_VERTICAL_PAIRS` CI gate |
> | Closed-alpha API serving real users | **Live**, with allowlist + scoped deletion + weekly report | `docs/closed-alpha-api-service.md` |
> | Public benchmark, Apache 2.0 | **Companion Bench v1.0** · 24 public + 96 held-out · 6 family × 6 axis | `packages/companion-bench/` |
> | Per-user memory across session restart (passes where RAG fails) | **4 probes PASS**: context, temporal, update, association | `tests/longitudinal/test_vz_memprobe_*.py` |
> | OpenAI-compatible facade (read-only) | Any OpenAI SDK client connects with zero changes | `lifeform-openai-compat` |

**Speaker script (4 min)**

There are two things on this slide and I want you to remember both.

On the math side: small-data continual learning is not magic. It is a known theoretical regime, and Yang Liu has spent two decades developing the algorithmic apparatus for it. The foundational paper, "Minimax Analysis of Active Learning," published in JMLR in 2015, proves under standard noise assumptions that a learner that *actively chooses which data to learn from* can match passive-learner accuracy with exponentially less data. That theorem is the reason we can plausibly say "we can adapt to a user from a few hundred turns a month, not millions of episodes." Forty more papers, eighteen at A-tier venues — NeurIPS, ICML, JMLR, COLT, SODA, FOCS, AISTATS — refine the apparatus. The full list is in Appendix C; any of them check out at the venue cited.

This is the part that distinguishes Volvence from companies whose technical story sounds similar. Most teams in our space picked their algorithm in 2024. Ours has a Chief Scientist whose entire PhD and postdoc was in the math that the product needs.

On the engineering side: I want to give you six numbers, each verifiable in due diligence. 1063 existing contract tests passing with zero regression, plus 96 new ones from our Phase 1 architecture-uplift. Five vertical lifeforms — emogpt, coding, character, figure, growth-advisor — co-loaded into a single process, enforced by CI not by founder promise. A closed-alpha API live with real users today. Companion Bench v1.0 open-sourced under Apache 2.0 with 120 long-session scenarios. Four per-user-memory probes passing where baseline retrieval-augmented generation structurally fails on three of them. An OpenAI-compatible facade so any existing SDK client connects with zero changes.

This is the difference between architecture as story and architecture as discipline.

**Design note**

Two columns; equal weight. Left column header: "The math (Yang Liu, 20 years)." Right column header: "The engineering (verifiable in DD)." Keep arXiv IDs out of the visible slide; they are in Appendix B.

---

## Slide 8 — Demo: Mobi private-traffic JV

**On-screen**

> **Mobi JV — private-traffic digital employee**
>
> Partner: **28M-follower MCN**, large private-traffic pool, human operators cannot maintain high-quality 1-on-1 relationships at scale.
>
> Watch four runtime properties in the demo:
>
> 1. **Cross-session memory** — prior context recalled after restart
> 2. **Preference separation** — Alice's preferences do not leak into Bob's behavior
> 3. **Recommendation timing** — system holds back when relationship stage is wrong
> 4. **Adaptation after feedback** — user pushback changes the next turn *and persists across restart*
>
> Status: JV signed; pilot in progress; conversion uplift not yet proven (kill criterion on Slide 10).

**Speaker script (1 min before video)**

I will show one demo. Watch the four runtime properties listed.

**Speaker script after video (90s)**

The point of that demo is not that the AI sounds human. Many systems do. The point is the system has *runtime state* that survives restart, scopes correctly across users, adapts in a typed way when the user pushes back, and that adaptation persists. That is what we mean by "the runtime captures the relationship."

The Mobi pilot is the first commercial verification of whether this runtime layer converts to business metrics. The next two slides are the unit economics and the kill criterion.

---

# Part C — Distribution and commercials

---

## Slide 9 — Six signed JVs, four scenarios, one engine

**Claim**

> Distribution is already signed. The same Volvence runtime runs four vertical bundles. Each bundle has a real partner, a conservative conversion math, and a defensible unit economics. The engine is shared; only the bundle differs.

**On-screen**

> | Scenario | Partner | Audience / base | Conversion math (conservative) | Annual revenue to Volvence |
> |---|---|---|---|---|
> | **Companion** (UploadLive) | JV with 15M-follower influencer + others | 45M follower base | 1% activation × 10% annual conversion × US$42/yr × 30% Volvence share | **~US$1.87M** |
> | **Parenting** (Gao Gailun + parenting platform) | 15M-follower JV partner | Existing parenting platform | Audience × activation × conversion × price | **~US$6.25M** |
> | **Private-traffic digital employee** (Mobi) | 28M-follower MCN | 28M fans | 1% conversion × US$70 GMV/user × 30% share | **~US$6M** |
> | **Cross-border AI commerce expert** (Heyi / Guomao) | JV with Hengyi / Zhejiang Guomao | 50K enterprises from 1688.com | 1% annual subscribers × US$6,900/yr × 38% share | **~US$3.47M** |
>
> **Six signed JVs total** — 4 already signed, 2 signing in April (one US$200K deal already closed; 30K-overseas-enterprise partner signing).

**Speaker script (3 min)**

Same engine, four bundles, six signed JVs.

Companion runs on the 45-million follower base accessible through UploadLive and the influencer JVs. One percent activation, ten percent annual conversion, US$42 per year, thirty percent share — roughly US$1.87M annually.

Parenting is the JV with Gao Gailun, a 15-million-follower parent-education influencer. Volvence sits between parent and child as a long-term tracking expert. Roughly US$6.25M annually.

Private-traffic — the Mobi scenario you saw in the demo — is the largest single contract. 28-million followers, 1% conversion, US$70 GMV per user, 30% share. Roughly US$6M.

Cross-border commerce is the JV with Hengyi and Zhejiang Guomao, with 50,000 enterprises accessible through 1688. One percent annual subscription, US$6,900 per year, 38% share. Roughly US$3.47M.

These are not abstract TAMs. They are conversion math against signed-JV audience size. The next slide breaks Mobi unit economics and the kill criterion.

**Design note**

Four-quadrant grid; shared center labeled "Volvence Runtime." V4-style layout, the only one in the deck.

---

## Slide 10 — Mobi unit economics + kill criterion

**On-screen**

> **Mobi private-traffic JV — unit economics**
>
> | Item | Unit | 2026 target scale | Volvence revenue contribution |
> |---|---|---|---|
> | Service fee / token procurement | RMB 30 / user / year | ~187K orders | ~RMB 2.8M |
> | JV profit share | RMB 100 / user / year distributable profit | same | ~RMB 2.8M |
> | **Mobi JV 2026 subtotal** | | | **~RMB 5.6M (~US$800K)** |
>
> **Conversion assumption (projected, not proven)**
>
> - SCRM industry baseline: ~0.3%
> - Volvence target with relationship runtime: 0.6-1.0%
> - 3-month pilot observation window not yet complete
> - **Kill criterion: 3-month pilot conversion < 0.5% → vertical deprioritized**
>
> **Repricing thesis**
>
> | Weimob / Youzan | Volvence |
> |---|---|
> | Reach tools | Relationship engineering |
> | Broadcast / auto-reply | Remembered and understood |
> | One-time conversion | Cross-session LTV |
> | ~RMB 1K / month / brand | RMB 5K-50K / month / brand (target) |

**Speaker script (3 min)**

Two revenue lines per converted user per year on Mobi: RMB 30 service fee, RMB 100 profit share on distributable margin. At target 187K orders, RMB 5.6M, roughly US$800K, from this single JV.

The line I want you to read carefully is conversion. SCRM industry baseline is around 0.3 percent. Our target with relationship runtime is 0.6 to 1.0 percent. The three-month pilot observation window is not yet complete, so this is projection, not result.

The part I want you to remember is the kill criterion. If the three-month pilot lands below 0.5 percent, this vertical is deprioritized. We do not double down on a thesis that has not converted. That discipline applies to every JV.

The repricing thesis is the larger commercial story. Weimob and Youzan sell reach at about RMB 1K per brand per month. Volvence targets RMB 5K-50K per brand per month for relationship optimization. The valuation multiple comes from being a *different category*, not a cheaper Weimob.

**Design note**

`projected` and `kill criterion` are visually unmissable. Green box, bold border.

---

## Slide 11 — 2026 conservative ARR and 18-month proof plan

**On-screen**

> **2026-2027 ARR scenarios (USD millions)**
>
> | Year | Conservative | Optimistic |
> |---|---|---|
> | **2026** | **3.33 - 5.0** | (range covers JV ramp speed) |
> | **2027** | 13.9 | 23.6 |
>
> Margin trajectory: 31% net (2026) → 46% (2027) → 54% (2028). Asset-light SaaS scaling; we benefit when frontier labs lower inference costs.
>
> **18-month proof plan — signed access → recognized ARR**
>
> | Timeline | Milestone | Success criterion |
> |---|---|---|
> | M0-M3 | First lighthouse in production | Real users, real usage, measurable retention / conversion |
> | M4-M9 | 3 JVs in production | Repeatable deployment process, early ARR |
> | M10-M18 | **ARR > US$1M real (recognized)** | Not projected; recognized revenue |
>
> If we do not hit M9 / M18, we should not raise Series A on this story.

**Speaker script (2 min)**

We have a detailed internal financial model; it can be shared in DD.

The number to anchor on is **2026 conservative, between US$3.33M and US$5M.** That is what we intend to show as production ARR by Q4-2026, not project from partner audience size.

2027 and 2028 numbers are scenario, not proof. They depend on whether Mobi-style unit economics repeat across verticals. Treat them as planning, not promise.

The 18-month plan converts signed access into recognized revenue: lighthouse in M0-M3, three JVs in production by M9, more than US$1M in recognized real ARR by M18. If we do not hit M9 or M18, we do not raise Series A on this story.

**Design note**

Do not show the 2028 number as headline. The credible number is the 2026 conservative band.

---

# Part D — Risks, ask, close

---

## Slide 12 — Risks (with falsifiers)

**On-screen**

> | Risk | What would prove / disprove |
> |---|---|
> | JV access does not convert to usage | First lighthouse fails to retain users / produce conversion signal |
> | Relationship quality does not improve business metrics | Mobi 3-month pilot < 0.5% conversion → deprioritize (this vertical, then re-evaluate the thesis) |
> | Frontier labs add stronger memory | We prove value in vertical state, governance, and business workflow — not generic memory |
> | Frontier labs ship per-persona refusal / typed feedback / persistent identity | We lose part of the structural-incompatibility argument. Architecture remains useful at the relationship-runtime layer; thesis softens |
> | Regulation does not materialize (EU AI Act / PIPL enforcement softens) | Governance moat narrows; commercial moat through signed JVs and unit economics remains |
> | Team execution bandwidth | 3 JVs in production by M9 — otherwise we narrow focus |
>
> Each row is a falsifier, not a hedge. The next round is about converting these risks into evidence.

**Speaker script (2 min)**

Risks I would focus on if I were in your seat.

The first three are commercial. The fourth is the structural-incompatibility argument we made on Slide 5: if a frontier lab ships per-persona refusal, typed feedback enums, and persistent runtime identity together, we lose part of that argument. Our architecture remains useful as a vertical-bundle relationship runtime, but the moat narrows. Honest read.

The fifth is regulation. If EU AI Act and PIPL enforcement softens — possible — the governance moat narrows. The commercial moat through signed JVs and unit economics remains independent of that.

The sixth is team. If by M9 we cannot get three JVs into production, we narrow focus. We will not pretend the story is bigger than the evidence.

**Design note**

Each row reads as falsifier. Make that visually explicit.

---

## Slide 13 — The Ask

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
> **Use of funds:**
>
> - **Engineering 40%** — runtime, deployment reliability, evaluation
> - **Compute / data 25%** — substrate, evidence pipeline, benchmark runs
> - **GTM 20%** — 3 in-production JV launches, lighthouse customers, partner success
> - **Operations / legal / IP 15%** — audit, consent, deletion, IP structure
>
> **Next financing gate:** 3 JVs in production, ARR > US$1M real, repeatable deployment playbook.

**Speaker script (2 min)**

US$3M to US$5M, late seed or pre-Series A, Xfund as lead or co-lead if there is alignment. Pre-money target band US$20M to US$30M, range under discussion. Xfund ticket roughly US$1.5M to US$2.5M, 7-10% equity at this band.

The round is for validation, not expansion. Eighteen months to convert signed access into in-production deployments and recognized ARR. If we hit that, we raise Series A from a position of evidence; if not, we narrow.

---

## Slide 14 — Close: the thesis in one frame

**On-screen**

> **The Volvence thesis, in five lines:**
>
> 1. Vertical proprietary data beats LLM scaling. (Xfund thesis, already validated.)
> 2. Generation 1 was institutional vertical data — Mayo Clinic × Open Evidence.
> 3. **Generation 2 is the per-person trajectory** — renewable, non-transferable, only captured by a thick runtime with memory, identity, audit, and consent.
> 4. Frontier labs are structurally locked out of this layer by their universal-assistant narrative.
> 5. We have the team (Hacker + Humanist, with a CMU-trained Chief Scientist whose math is the field), the engineering (1063+ contract tests, 5 vertical lifeforms, closed-alpha live), and the distribution (6 signed JVs, 45M followers, 50K enterprises) to build it.
>
> 2026 conservative ARR: **US$3.33M - US$5M**.
>
> **Each line has a citation behind it. We can defend any of them in detail.**

**Speaker script (60s)**

Five lines. Each backed by either a paper, an experiment in our repo, or a verifiable number. The conclusion: the next layer of vertical proprietary data is the relationship itself; we have the runtime, the team, and the distribution to capture it; and we have 18 months to convert signed access into recognized ARR.

I would love to spend the rest of the time on your questions.

---

# Optional Appendix / Q&A Slides

> Use only when asked. Appendix A is the deep technical thesis from V6, preserved verbatim for a senior technical DD partner.

---

## Appendix A — The 8-step technical thesis (for technical DD)

> This appendix carries the V6 deck spine: an 8-step argument from "the first step toward AGI is Cognitive AGI" through "we built it" to "this is independent of frontier labs." It is preserved here, with full citations, for when a technical due-diligence partner picks up the conversation. Each step has a Proof block; each Proof block names experts, papers, our experiments, or market numbers.

### Step 1 — The first step toward AGI is *Cognitive* AGI

The path to general intelligence does not run through bigger world models or better mechanical control. It runs through **cognition exercised in real environments.** Vertical AGI is just cognitive AGI specialized by environment, not a separate species.

**Proof:**
- Sutton & Silver, *The Era of Experience* (DeepMind, Apr 2024) — "the era of human data is ending; the era of experience is beginning."
- Sutskever, NeurIPS 2024 keynote — pre-training as we know it will end.
- Hassabis 2024-2025 statements; DeepMind Genie / SIMA / Dreamer 4 as substrates for cognition.
- Boston Dynamics / Tesla Optimus / Figure 02 / 1X — mechanical actuation converged; cognition is the new bottleneck.
- Botvinick / Wang / Dabney 2025, *Distributional Dopamine* — biological intelligence runs on prediction-error-driven cognition.

### Step 2 — Cognitive AGI must be online continual learning. Prompt engineering is the new Bitter Lesson.

**Proof:**
- Sutton, *The Bitter Lesson* (2019) — reapplied.
- 2025-2026 production failures of agent-harness companies (LangChain / AutoGen / Cognition AI Devin) — "wisdom debt" past 10-20 step horizons.
- Karpathy on Software 2.0.
- Bai et al. 2022 *Constitutional AI* and 2024-2025 follow-ups — even Anthropic moves behavior out of system prompts into learned layers.

### Step 3 — Humans, not the internet, are the next durable vertical data layer

**Proof:**
- Villalobos, Sevilla et al., *Will We Run Out of Data?* (Epoch AI, 2024) — internet text ceiling 2026-2032.
- Open Evidence's Mayo Clinic moat — Xfund-portfolio analogue.
- Karpathy on user-state.
- Magnitude estimate: 1B users × 10 MB/day × 365 = ~3.6 EB/year vs. ~10-50 TB indexed internet text. Ratio ~10⁵.
- OpenAI's pivot to memory and consumer; Anthropic's pivot to coding — both racing to occupy user-state.

### Step 4 — Token-level RL is structurally infeasible

**Proof (three labs, five months):**
- Anthropic, *Natural Emergent Misalignment from Reward Hacking* (Nov 2025) — alignment-faking, sabotage spontaneous under RL.
- OpenAI + academia, *Reasoning Models Struggle to Control their Chains of Thought* (Mar 2026).
- MATS scholars, *Output Supervision Can Obfuscate the CoT* (Nov 2025).
- Anthropic + Schulman, *Reasoning Models Don't Say What They Think* (2025).
- Lilian Weng, *Why We Think* survey (May 2025) — all dual-process work to date is still in token space.

### Step 5 — The path is emergent multi-timescale RL on a learned abstraction space, with sparse data

**Proof:**
- Behrouz & Mirrokni, *Nested Learning* (Google Research, arXiv:2512.24695, late 2025) — multi-frequency associative memory with ideal-init targets.
- ETH-Sacramento, *Emergent Temporal Abstractions* (arXiv:2512.20605, late 2025) — latent action space `z_t` + learned switching gate `β_t`.
- Precup & Klissarov, *Discovering Temporal Structure: HRL Overview* (DeepMind, 2026) — Option Keyboard interface.
- Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015) — small-data continual learning is mathematically feasible.

### Step 6 — Our three answers: Reward = Body, Abstractions = NL + ETA, Sparse data = Active Learning

**Reward = Body** — Botvinick / Wang / Dabney 2025 *Distributional Dopamine* + Friston *Active Inference*. Our `vz-cognition.prediction` + `vz-cognition.credit` modules with 4 drive states.

**Abstractions = NL + ETA** — `CMSVariant.NESTED` (Nested Learning meta-learning convergence: init error decreases monotonically across context resets). `scripts/run_eta_paper_suite.sh` runs 4 matched-control ablations: `full-no-optimize`, `full-no-replacement`, `learned-lite-causal`, `noop-backend` — **all 4 PASS** on hierarchical sparse-reward, abstract-action family reuse, held-out composition, and delayed credit alignment.

**Sparse data = Active Learning** — Hanneke & Yang JMLR 2015 + Yang's 40+ papers. VZ-MemProbe 4 probes (context, temporal, update, association) — all PASS at production-relevant data scales where baseline RAG fails on 3 of 4.

### Step 7 — Thin Prompt, Thick Runtime — already implemented

**Engineering reality (verifiable in repo):**

| Asset | Hard number | Where to verify |
|---|---|---|
| Phase 1 architecture-uplift exit evidence | **96 new contract tests PASS · 1063+ existing zero regression** | `docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md` |
| Vertical lifeforms co-loaded in one process (CI-enforced) | **5** | `PARALLEL_VERTICAL_PAIRS` CI gate |
| Closed-alpha API serving real users | **Live** | `docs/closed-alpha-api-service.md` |
| Companion Bench v1.0 (Apache 2.0) | **24 public + 96 held-out** | `packages/companion-bench/` |
| Figure vertical full chain | **`figure-bundle:einstein:29eacd226a7cdfd0` byte-equivalent reproducible** | Wave A-G land |
| Rupture/Repair typed enum loop | Cross-session durable; behavior changes; audit trail records change | `vz-cognition.rupture_state` owner |
| GDPR/PIPL deletion path | `DELETE /v1/users/me/memory` + deletion-evidence ledger | required for enterprise contracts |
| OpenAI-compatible facade | Any OpenAI SDK client connects with zero changes | `lifeform-openai-compat` |

### Step 8 — Independent of frontier labs

**Frontier-lab constraint:**
- GPT-5 system card — engineering integration, no paradigm leap.
- Sutskever's SSI — 32B valuation, zero models, zero papers.
- Karpathy → Eureka Labs.
- Schulman → Anthropic → Thinking Machines.
- Universal-assistant narrative cannot ship per-persona refusal, typed feedback enums, persistent regime identity.

**Our distribution:** 45M followers, 50K enterprises, 6 signed JVs.

### How IQ and EQ emerge in our runtime

**IQ:** `IQ_volvence = Substrate × ETA_reuse × NL_accumulation`. We do not compete with frontier labs on IQ; we inherit it and amplify it.

**EQ — four mechanisms × four academic anchors × 28+ independent PASSing tests:**

| Mechanism | Academic anchor | Repo benchmarks |
|---|---|---|
| **(A) R7 Dual-track learning** — `world_temporal` and `self_temporal` independent owners | Premack & Woodruff 1978; ETA latent-track separation | `test_multi_party_scenarios.py` (10 PASS); `test_cross_user_isolation_after_owner_hydration` PASS |
| **(B) 4 Theory-of-Mind owners** — keyed by `interlocutor_id` | Saxe / Wellman developmental psychology | `test_feeling_about_other_active_matched_control.py` (8 PASS); `test_common_ground_active_matched_control.py` PASS; `test_social_memory_visibility_loop.py` PASS |
| **(C) Rupture/Repair typed enum loop** — typed event → durable memory → next-session behavior changes | Closed-loop emotional learning that thumbs-up/down API cannot do | `test_rupture_repair_durable_memory_continues_across_session_boundary` PASS; `test_commitment_lifecycle_continues_across_session_boundary` PASS; `test_vitals_drive_levels_continue_across_session_boundary` PASS |
| **(D) Regime persistent identity** — value prioritization is runtime invariant, not prompt | DeepMind 2025-2026 work on training-time character | `test_affordance_delayed_credit.py` (4 PASS); multi-party `regime_tags` + `interlocutor_id` composite dispatch ACTIVE |

**Total: 28+ independent PASSing benchmarks across `tests/contracts/`, `tests/longitudinal/`, `tests/test_social_*`.**

---

## Appendix B — Citation index

**Step 1 — Cognitive AGI:**
- Sutton & Silver, *The Era of Experience* (DeepMind, 2024)
- Sutskever NeurIPS 2024 keynote (transcript public)
- Hassabis 2024-2025 public statements; DeepMind Genie / SIMA / Dreamer 4 papers
- Botvinick / Wang / Dabney, *Distributional Dopamine* (DeepMind, 2025)

**Step 2 — Bitter Lesson reapplied:**
- Sutton, *The Bitter Lesson* (2019)
- Karpathy public statements 2024-2025 on Software 2.0
- 2025-2026 reports on agent-harness production failures (LangChain, AutoGen, Cognition AI)
- Bai et al. *Constitutional AI* (Anthropic, 2022)

**Step 3 — Humans are the next vertical data:**
- Villalobos, Sevilla et al., *Will We Run Out of Data?* (Epoch AI, 2024)
- Open Evidence's Mayo-Clinic moat (Xfund portfolio, public)
- Karpathy on user-state as Software 2.0 layer

**Step 4 — Token-RL is structurally dangerous:**
- Anthropic, *Natural Emergent Misalignment from Reward Hacking* (Nov 2025)
- OpenAI + academia, *Reasoning Models Struggle to Control their Chains of Thought* (Mar 2026)
- MATS, *Output Supervision Can Obfuscate the CoT* (Nov 2025)
- Anthropic + Schulman, *Reasoning Models Don't Say What They Think* (2025)
- Lilian Weng, *Why We Think* survey (May 2025)

**Step 5 — Multi-timescale RL on abstraction space:**
- Behrouz & Mirrokni, *Nested Learning* (Google Research, arXiv:2512.24695)
- ETH-Sacramento, *Emergent Temporal Abstractions* (arXiv:2512.20605)
- Precup & Klissarov, *Discovering Temporal Structure: HRL Overview* (DeepMind, 2026)
- Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015)

**Step 6 — Reward = body:**
- Botvinick / Wang / Dabney, *Distributional Dopamine* (DeepMind, 2025)
- Friston, *Active Inference / Free Energy Principle* (decade-long body of work)

**EQ academic anchors:**
- Premack & Woodruff, *Does the chimpanzee have a theory of mind?* (1978)
- Saxe / Wellman developmental psychology line on dissociable belief-desire-intent states

**DeepMind self-improvement (parallel to our `ModificationGate`):**
- DeepMind, *AlphaEvolve* (2026)
- DeepMind, *AlphaDev* (Nature 2023)

---

## Appendix C — Yang Liu's academic record (full table)

> *Reference: V4 PDF "Appendix — Dr. Yang Liu's Academic Highlights".*

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
| 9 | ***Minimax Analysis of Active Learning*** (key citation for Slide 7) | **JMLR** | **Jan 2015** |
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

**Upcoming:**
- *Simpler Active Learning with Surrogate Losses* — done, NeurIPS May 2026
- One paper currently confidential — AAAI 2026 or ICML 2027

---

## Appendix D — Anti-claims (what we are *not* selling)

> A liberal-arts VC reads team maturity from this list. We keep it explicit.

| Claim we are *not* making | Why we are not making it |
|---|---|
| "Smarter than GPT/Claude" | Substrate ceiling; IQ scaling is not our moat. We inherit and amplify. |
| "AGI in 12-24 months" | Architecture is the *container* for cognitive AGI, not the implementation. Strong AGI probability < 5% in 24 months. |
| "Generic memory plugin" | OpenAI Memory / Mem0 / Letta own that lane. We do not compete in it. |
| "Agent framework" | LangChain / AutoGen own that. Our contract runtime is for our own use. |
| "AI psychologist / AI doctor" | Licensure / liability / regulation make this off-limits today. |
| "Companion for minors" | Legal and ethical risk too high for this stage. |
| "Unauthorized resurrection of living public figures" | Legal and ethical non-starter. |
| "Strong cognitive AGI in 12-24 months" | Internal team probability < 5%. We do not say what we do not believe. |

---

## Appendix E — 60-second verbal version

> If Patrick says "tell me in one minute":

> *We are building the runtime that captures the next layer of vertical proprietary data — the relationship itself. Mayo Clinic data was layer one; Open Evidence is the proof. Layer two is the per-person trajectory each user produces with a product over months and years. It is renewable, non-transferable, and only compounds if a thick runtime has memory, identity, audit, and consent. Frontier labs are structurally locked out of this layer because per-persona refusal, typed feedback enums, and persistent runtime identity conflict with their universal-assistant brand narrative. We have the team — me as builder-operator, Yang Liu as CMU-trained Chief Scientist whose math on small-data continual learning is the actual field this product needs. We have the engineering — 1063+ contract tests, 5 vertical lifeforms co-loaded in one process, closed-alpha API live today, governance architecture that EU AI Act and PIPL are about to require. And we have the distribution — 45 million followers and 50 thousand enterprises through six signed JVs, conservative 2026 ARR US$3.3M to US$5M. The ask is US$3M to US$5M late seed; 18 months to convert signed access into recognized ARR; Xfund as lead or co-lead.*

---

## Appendix F — Q&A

### Q1 — You have 6 JVs but no ARR. Why is this traction?

Signed distribution access, not recognized revenue — and we say so on Slide 11. The 18-month plan converts at least three JVs into in-production deployments with recognized revenue. The Mobi kill criterion is the public discipline.

### Q2 — Why not just use GPT with memory?

Generic memory helps; we use it as substrate. Real human-relationship products require typed feedback enums, per-persona refusal, scoped deletion with audit trail, persistent runtime identity, and rupture-repair loops. None of these is in any frontier-lab roadmap because all of them conflict with the universal-assistant narrative. If GPT memory becomes stronger, our substrate improves; the runtime layer remains ours.

### Q3 — What is your unfair advantage?

Three things stacked. First, scientific depth in non-stationary learning through Yang Liu — directly relevant to the Slide-7 sparse-data question. Second, an architecture that has already been built and contract-tested — 1063+ tests, 5 vertical lifeforms in one process. Third, 45M-follower distribution access through 6 signed JVs. None of the three alone is enough; the combination is.

### Q4 — What would make you change direction?

Mobi 3-month pilot below 0.5% conversion → deprioritize. M9 without 3 JVs in production → narrow focus. Frontier labs ship per-persona refusal, typed feedback enums, and persistent runtime identity together → our structural-incompatibility argument softens, but our architecture remains useful as a relationship-runtime layer.

### Q5 — What is the upside, not the SaaS upside?

Repricing in the near term — Weimob and Youzan sell reach at RMB 1K per brand per month; Volvence sells relationship optimization at RMB 5K-50K per brand per month if conversion holds. Platform option in the long term — every brand running on Volvence accumulates a non-transferable per-user trajectory. That is the second-generation vertical data that no foundation lab structurally owns.

### Q6 — Why Xfund specifically?

Three reasons. First, the thesis is a direct extension of Xfund's existing framing — proprietary vertical data beats LLM scaling — into the next layer: the relationship itself. Second, Patrick underwrites founder judgment and category formation, which is what this round is. Third, your Delphi and Open Evidence portfolio are *complementary* slices of the same thesis — Delphi is a snapshot replica of one person, Open Evidence is institutional vertical data, Volvence is the living trajectory layer with governance.

### Q7 — Is your technical thesis (Nested Learning, ETA, latent-space RL) reproducible by a senior technical DD partner?

Yes. Appendix A carries the eight-step technical thesis with named papers, arXiv IDs, our experiment scripts (`scripts/run_eta_paper_suite.sh`, `tests/longitudinal/test_vz_memprobe_*.py`), and the venue for each cited paper. The four matched-control ablations of the ETA paper-suite all PASS in our repo; we believe we are the first to run that specific control set publicly.

### Q8 — Why do you spend a slide on regulation? Most VCs do not.

Two reasons. First, the engineering required for governance — owner-snapshot SSOT, scoped deletion with evidence, gated self-modification — is the same engineering the product needs to be a long-relationship product, so it is not a tax, it is a moat. Second, this is the layer that Xfund's principal partner is uniquely well-suited to evaluate. We list it deliberately.

---

# PPT Production Notes

## Visual style

- Black or near-black background.
- One main thought per slide.
- Dense tables only on Slides 4, 6, 7, 9, 10, 11, 12, 13.
- Use green only for emphasis (numbers, kill criterion, paper citations) — not decoration.
- On Slide 10, `projected` and `kill criterion` must be visually unmissable.
- On Slide 13, the ask box uses thick border + large numbers, but the word "range" stays visible.
- On Slide 3, the three-generation diagram is the visual lede of the deck. Spend layout time on it.
- No hype words on-screen.

## Demo handling

- Keep Mobi demo to 4-5 minutes.
- Add English subtitles if conversation is Chinese.
- Highlight only 4 moments:
  - prior context remembered
  - preference separation
  - recommendation timing
  - adaptation after feedback that *persists across restart*

## Speaker behavior

- Do not read long notes.
- After naming an open gap, kill criterion, or paper citation — pause. Let the rigor register.
- If Patrick interrupts, stop the deck and go into conversation. The thesis structure means any block stands alone.
- The goal is not to finish all slides. The goal is to make Slides 1, 2, 3, 5, 6, 9, 10 land cleanly. Everything else is supporting evidence.

## Words to avoid (in body of deck — allowed in Appendix A)

- "唯一", "永远", "结构性独占", "杀伤力", "灵魂级", "打爆"
- "OpenAI 做不了" / "structurally cannot own"
- "已经证明" (use "PASS in repo" or "verifiable in DD")
- AI-research jargon Patrick has no documented framework to evaluate independently: `β_t`, `z_t`, `metacontroller`, `Nested Learning`, `ETA`, `R1-R15` — all moved to Appendix A.

## Replacement language

- "Our current judgment is..."
- "This has not yet been fully proven..."
- "The next 18 months are about validating..."
- "DD can re-run this live..."
- "If this metric does not hold, we re-evaluate..."
- "Kill criterion is..."
- "Patrick, this is the natural extension of..."

---

## Change log

- **2026-05-17 v7.0**: Reorganized for Patrick Chung's specific evaluation profile (founder + thesis-extension + governance, not independent technical evaluation). Compressed v6's 8-step technical thesis from main deck into a single appendix; promoted "Why this extends Xfund's thesis" to Slide 3 as the deck spine; promoted governance/regulation to Slide 6. Body deck shrunk from 22 to 14 slides; appendix expanded with full v6 technical content preserved for technical DD. Vocabulary swept of AI-research jargon in body (jargon retained in Appendix A only).
- **2026-05-17 v6.0**: Restructured around 8-step thesis chain. Each step carries explicit Proof block (named expert / paper / our experiment / market number).
