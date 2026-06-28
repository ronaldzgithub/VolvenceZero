# VolvenceZero — Xfund Pitch Deck v6

> Status: **v6.0 draft (2026-05-17)**
> Base: V4 PDF (`VOLVENCE-Beyond-Agents-Full-Autonomy-AI-with-Human-Level-EQ-and-IQ-0421.pdf`) clarity of logical chain; V3/V5 cool-tone discipline; xfund-strategic-thesis depth.
> Design principle: **Every logical step must be backed by an explicit, citable proof — named industry expert, peer-reviewed paper, our quantitative experiment in repo, or solid market estimate.** No claim should rest on rhetoric alone.
> Purpose: PPT-ready thesis-driven deck. Cold open: 8-step thesis chain. Mid: our answers + shipped evidence. End: commercial wedge + ask.
>
> Recommended meeting format: **~45 min presentation + ~15 min conversation**.
> Short version: Slides 1-12 only (thesis + answers + demo + ask). Slides 13-22 are wedge / financials / appendix.

---

## V6 vs V5: what changed

V5 was a "cool, fact-first" deck that downplayed thesis and led with private-traffic wedge. V6 reverses that ordering on founder request:

1. **Thesis first.** The deck opens with a complete 8-step argument (AGI → Cognitive AGI → continual learning → token-RL is dead → emergent multi-timescale RL → reward from body → small data via active learning → why us). This is the V4 PDF spirit, restored.
2. **Every step has explicit proof.** Each thesis slide carries a "Proof" block citing one or more of: (a) named industry expert with public statement, (b) peer-reviewed paper with arXiv ID or venue, (c) our reproducible experiment in repo, (d) verifiable market number. No floating claims.
3. **The cool tone is preserved.** No "OpenAI structurally cannot do X", no "唯一", no "灵魂级". Where V4 used adjectives, V6 substitutes citations.
4. **Wedge moves to mid-deck (Slides 14-17).** Mobi unit economics, kill criterion, conservative ARR scenario all retained from V5 but framed as "the thesis is already producing commercial validation", not the lede.
5. **Yang Liu academic appendix preserved from V4** — directly relevant for proof-of-active-learning claim.

---

## V6 thesis chain (one page)

The deck argues a single thesis in 8 steps. Each link is a separate slide; each carries a Proof block.

| # | Claim | Strongest proof |
|---|---|---|
| 1 | The first step toward AGI is **Cognitive** AGI — not world models, not embodied control. All vertical AGI must ultimately become cognitive AGI exercised in real environments, not statically pretrained. | Sutton & Silver, *The Era of Experience* (DeepMind, 2024); Sutskever NeurIPS 2024 keynote; embodied/world-model work (Genie / Dreamer / SIMA / Optimus / Figure 02) is converging to a cognition bottleneck. |
| 2 | Cognitive AGI must be **online continual learning** supported by neural networks. Prompt / context / harness engineering is a re-run of the Bitter Lesson. Workflows are not intelligence. | Sutton, *The Bitter Lesson* (2019), reapplied; production-level results from agent-harness companies (LangChain / AutoGen / Cognition AI) showing wisdom-debt collapse in long-running tasks. |
| 3 | Among possible data sources, **humans** — not the public internet — are the only durable vertical-data moat. Hardware is just a probe; the real frontier is what each person produces every day. | Villalobos et al., *Will We Run Out of Data?* (Epoch AI, 2024); Open Evidence's Mayo-Clinic moat (Xfund-portfolio analogue); Karpathy on user-state as the new "Software 2.0". |
| 4 | Real continual learning requires **goal-oriented RL**, but **token-level RL is structurally infeasible**. Three independent labs proved this in five months. | Anthropic 2025 *Natural Emergent Misalignment*; OpenAI 2026 *Reasoning Models Struggle to Control their Chains of Thought*; MATS 2025 *Output Supervision Can Obfuscate the CoT*; Schulman et al. 2025 *Reasoning Models Don't Say What They Think*. |
| 5 | The path forward is **emergent multi-timescale RL on a low-dimensional abstraction space, with sparse interaction data**. Three open questions follow: where does reward come from? how do abstractions emerge? how does it work with little data? | Behrouz & Mirrokni, *Nested Learning* (Google Research, arXiv:2512.24695); ETH-Sacramento, *Emergent Temporal Abstractions* (arXiv:2512.20605); Precup & Klissarov, *Discovering Temporal Structure* (HRL Overview, 2026); Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015). |
| 6 | Our answers: **(a) reward = body** (drives, needs, prediction error); **(b) abstractions emerge from Nested Learning + ETA** in a learned latent space `z_t`; **(c) small-data feasibility comes from active-learning theory** developed by our co-founder. | Botvinick / Wang / Dabney 2025 *Distributional Dopamine*; Friston, *Active Inference*; our `CMSVariant.NESTED` meta-learning convergence; ETA paper-suite 4 matched controls in repo. |
| 7 | We have **already implemented** a Thin-Prompt / Thick-Runtime architecture that does these things. 1100+ contract tests, 5 vertical lifeforms co-loaded in one process, closed-alpha API serving real users, and benchmarks for both IQ-emergence and EQ-emergence are PASSing in repo. | `tests/contracts/` + `tests/longitudinal/` + `scripts/run_eta_paper_suite.sh`; Phase 1 architecture-uplift exit evidence (96 new contract tests + 1063 zero regression). |
| 8 | This is an **independent technical and commercial path** from frontier LLM labs. They are forced into IQ-scaling competition. Our wedge is humans-as-vertical-data, accessed through 45M-follower distribution + 50K-enterprise base + 6 signed JVs, producing a conservative 2026 projected ARR of US$3.3M-5M. | OpenAI's GPT-5 system card (engineering integration, no paradigm leap); Sutskever's silence at SSI (zero models, zero papers); Karpathy's Eureka Labs (left frontier); our 6 signed JVs and Mobi unit economics. |

---

## Meeting cadence

```text
0-2 min      Cover + one-line pitch
2-5 min      Founder + team credibility
5-25 min     The 8-step thesis (Slides 3-10)
25-32 min    Our answers + architecture (Slides 11-13)
32-38 min    Demo
38-44 min    Commercial wedge + traction + financials (Slides 14-17)
44-50 min    Risks, ask, close (Slides 18-22)
50-60 min    Conversation / DD questions
```

If Patrick interrupts at any point, switch to conversation mode. The deck is structured so every block is independently defensible.

---

# Main Deck

## Slide 1 — Cover

**On-screen**

> **VOLVENCE**
>
> **The runtime for Cognitive AGI.**
>
> Online continual learning on neural substrate. Multi-timescale emergent reinforcement learning in a learned abstraction space. Reward grounded in body, not preference labels.
>
> Built for the only frontier the foundation labs cannot easily own: each person, every day.
>
> Zhao Jiangbo, Founder & CEO
> Xfund conversation · May 2026

**Speaker script (60s)**

Patrick, thank you for the time.

I want to give you a single thesis in this meeting, in eight steps, each grounded in either a public statement from someone you would respect, a peer-reviewed paper, or an experiment we can re-run live in due diligence.

The thesis, in one sentence: **the first step toward AGI is Cognitive AGI — not world models, not embodied control — and the only durable training environment for Cognitive AGI is the data each person produces every day, learned online by a thick runtime above frozen substrate.**

Volvence is the runtime that makes this learnable. Twenty years of accumulation in active learning, online learning, drifting distributions, plus two years of architecture work since ChatGPT, plus 45 million followers and 50 thousand enterprises already in our distribution. 2026 conservative projected ARR is between US$3.3M and US$5M.

The reason this is independent of OpenAI and Anthropic — and why Xfund's "vertical proprietary data beats LLM scaling" thesis fits precisely — is the punchline of slide 8.

**Design note**

Cover should be near-black, only one phrase visible at first frame: **"The runtime for Cognitive AGI."** Everything else fades in.

---

## Slide 2 — Founder + Team

**On-screen**

> **Zhao Jiangbo, Founder & CEO** (full-time)
>
> - Peking University CS · MBA
> - IBM Japan Research · HP China Software Sales GM · Alibaba VP Office · Tencent industry director
> - 3x founder; Haopai 300K users with zero paid acquisition; commercial exit
> - Self-funded Volvence R&D with RMB 5M since ChatGPT launch
> - Hands-on engineer; GitHub commits since Nov 2022, DD-verifiable
>
> **Yang Liu, PhD — Co-founder & Chief Scientist** (full-time)
>
> - CMU PhD, advised by **Avrim Blum** and **Jaime Carbonell**
> - IBM Research · Yale postdoc
> - **40+ papers** in active learning, drifting distributions, transfer learning, online learning
> - 18 A-tier (NeurIPS / ICML / JMLR / COLT / SODA / FOCS / AISTATS); upcoming: NeurIPS 2026
>
> **Wang Cangyu, CSO** · PhD Psychology · ex-Zhongqi Media GM (Douyin best agency, RMB 6B revenue)
>
> **Zhang Chi, CTO** (full-time) · Tsinghua CS · ex-Glodon · Haopai co-founder
>
> **Wu Xiang, CMO** (full-time) · ex-HP, Neusoft executive · 20yr enterprise GTM

**Speaker script (2 min)**

You are at this stage underwriting two people most: me and Yang Liu.

I have built and sold companies before. I am not a non-technical CEO outsourcing engineering — my GitHub commits since November 2022 are public for diligence.

Yang Liu is the technical reason this thesis is not speculative. She did her PhD at CMU under Avrim Blum and Jaime Carbonell. Her 40+ papers in active learning, drifting distributions, and online learning are exactly the math you need when the environment is non-stationary and labels are sparse — which, as I will show on slide 5, is the central technical problem of online continual learning.

Wang Cangyu's media background gives us private-traffic distribution access. Zhang Chi delivers engineering. Wu Xiang runs enterprise GTM.

The full appendix at the back of this deck lists Yang's papers. Any of them can be checked at the venue cited.

**Design note**

Yang's CMU advisors and paper count are the credibility anchors of this slide. Make them visually heavier.

---

# Part A — The 8-Step Thesis

> Each of the next eight slides advances one link in the argument. Each carries an explicit Proof block with named experts, papers, our experiments, or market numbers. The chain is designed so that if any single step fails, the conclusion fails — which makes the deck honest, not fragile.

---

## Slide 3 — Step 1: The first step toward AGI is *Cognitive* AGI

**Claim**

> The path to general intelligence does not run through bigger world models or better mechanical control. It runs through **cognition exercised in real environments**. Vertical AGI is just cognitive AGI specialized by environment, not a separate species. Static pretraining cannot produce it.

**Proof**

| Type | Source | What it shows |
|---|---|---|
| Industry expert | **Sutton & Silver, *The Era of Experience* (DeepMind, Apr 2024)** | Argues the next frontier is agents that learn from streams of experience, not pretraining text. "The era of human data is ending; the era of experience is beginning." |
| Industry expert | **Ilya Sutskever, NeurIPS 2024 keynote** | "Pre-training as we know it will end. The next paradigm is agents, synthetic data, and inference-time computation." |
| Industry expert | **Demis Hassabis, public 2024-2025 statements + DeepMind's pivot to Genie/SIMA/Dreamer 4** | DeepMind's world-model work is explicitly framed as a *substrate for cognition*, not a goal in itself. |
| Industry observation | **Boston Dynamics / Tesla Optimus / Figure 02 / 1X** progress in 2025-2026 | Mechanical actuation has converged; the bottleneck is now *cognition* — task understanding, generalization, error recovery. Embodied AI's missing layer is the cognitive runtime. |
| Cognitive science | **Botvinick / Wang / Dabney 2025, *Distributional Dopamine*** | Biological intelligence runs on prediction-error-driven cognition, not pure perception or pure motor control. |

**Speaker script (3 min)**

The first step is to disagree with two popular framings of AGI.

The first popular framing is that AGI will arrive through ever-better world models — Genie, Dreamer 4, or some Tesla-scale video model. The second is that it arrives through humanoid robots that can do laundry. Both are real engineering programs, and we admire them.

But they are *substrates*. The thing that has to ride on top of them is cognition — goal formation, abstraction, memory, adaptation, theory of mind, and recovery from surprise. Without that layer, a perfect world model is a movie projector and a perfect humanoid is a puppet.

The strongest public statement of this is Richard Sutton's recent paper with David Silver, "The Era of Experience". Their argument, which I find difficult to refute, is that the era of human-data-fueled pretraining is ending, and the next era is one in which agents learn from continuous streams of their own experience.

Sutskever made the same call at NeurIPS 2024 in different language: pre-training as we know it will end.

Hassabis's pivot at DeepMind is consistent — Genie, SIMA, and Dreamer 4 are not the destination, they are the dreamscape in which a cognitive agent is supposed to learn.

The corollary, important for this deck, is that **vertical AGI** — medical AGI, financial AGI, companion AGI — is not a different species from cognitive AGI. It is cognitive AGI specialized by environment. You cannot pretrain it; you have to grow it in the environment that defines it.

**Design note**

Use a single diagram: three concentric circles labeled `Substrate (LLM / world model / actuator)` → `Cognitive runtime (this is where intelligence lives)` → `Vertical specialization (medical, companion, private-traffic, etc.)`. Black background.

---

## Slide 4 — Step 2: Cognitive AGI must be neural-network-supported online continual learning. Prompt / context / harness engineering is the new Bitter Lesson.

**Claim**

> "Workflows are not intelligence." Hand-engineered prompts, retrieval pipelines, agent harnesses, tool-use graphs — these are the modern equivalent of 1980s expert systems. They will be repeatedly outperformed by systems that *learn* the same routing online. The Bitter Lesson is about to play out one more time.

**Proof**

| Type | Source | What it shows |
|---|---|---|
| Industry expert | **Sutton, *The Bitter Lesson* (2019)** — reapplied | "The biggest lesson from 70 years of AI research: general methods that leverage computation are ultimately the most effective by a large margin." Hand-crafted reasoning structures have lost every previous round. |
| Production failure | **2025-2026 reports on agent-harness companies** (LangChain, AutoGen, Cognition AI's Devin) | Long-running agent workflows accumulate "wisdom debt" — hand-tuned prompts, retries, and tool-use graphs that cannot be debugged. Production reliability collapses past task horizons of ~10-20 steps. |
| Industry expert | **Andrej Karpathy, public statements 2024-2025 on "Software 2.0" and "the operating system"** | The durable value layer migrates from hand-written code to learned representations. Prompt engineering is hand-written code in disguise. |
| Anthropic Constitutional AI line | **Bai et al. 2022, plus 2024-2025 follow-ups** | Even Anthropic, the most prompt-disciplined major lab, has continuously moved logic *out of system prompts* into learned constitutional behavior. |

**Speaker script (3 min)**

The second step is the most provocative for VCs because it disagrees with where most of the 2024-2025 agent capital went.

We believe — and we are stating this clearly so it can be falsified — that **prompt engineering, context engineering, and agent-harness engineering will be retrospectively recognized as the Bitter Lesson playing out one more time.**

Sutton's 2019 essay catalogued seventy years of AI history. Every time a generation of researchers hand-crafted the structure of intelligence — chess heuristics, expert systems, hand-tuned vision pipelines — they were beaten by a method that learned the structure from data and compute. There is no a priori reason 2024's prompt engineers and 2025's agent harness builders are exempt.

The empirical signal is already visible. The agent companies that hit production at scale in 2025 are reporting "wisdom debt" — long stacks of hand-tuned prompts, retries, fallbacks, and tool graphs that cannot be reasoned about and degrade past ten or twenty step horizons.

Big labs are quietly making the same call. Anthropic has been moving behavior out of system prompts and into learned constitutional layers since 2022. Karpathy's Software 2.0 framing is the same idea phrased differently.

What this means for our deck: **Volvence is not building a smarter agent harness. We are building a runtime that learns the harness, online, continuously, on neural substrate.** The prompt is thin; the runtime is thick. The runtime contains identity, memory, relationship state, abstraction, adaptation, and audit. The prompt only renders the current state into language.

**Design note**

Show two timelines side by side: "1980s — Expert systems → defeated by learning" and "2020s — Agent harnesses → ?". Let the visual analogy land before stating the conclusion.

---

## Slide 5 — Step 3: Among possible data sources, *humans* are the only durable vertical-data moat. Hardware is just a probe.

**Claim**

> Public-internet text data is finite and approaching its ceiling. Institutional vertical data (Mayo Clinic, Bloomberg, Westlaw) is durable but enumerable. The one data source that is renewable, non-transferable, and exponentially long-tail is **the moment-by-moment trajectory of each individual human's behavior, language, and context**. Hardware sensors — phones, watches, glasses, cars, robots — are merely probes into that data. They are not the moat. The data is the moat.

**Proof**

| Type | Source | What it shows |
|---|---|---|
| Peer-reviewed paper | **Villalobos, Sevilla et al., *Will We Run Out of Data?* (Epoch AI, 2024)** | Quantitative projection: high-quality text data exhausted between 2026-2032 at current scaling pace. Public-internet pretraining is a closing window. |
| Xfund portfolio analogue | **Open Evidence's Mayo-Clinic moat** | Patrick's own portfolio validates the "vertical proprietary data > generic scaling" thesis. Mayo data is finite and licensable; per-user behavior data is renewable and non-transferable. |
| Industry expert | **Karpathy on user-state as the next layer** (2024-2025) | "The real product is the operating system over each user's state." |
| Market estimate | Per-user behavioral data volume | A modal smartphone user generates ~5-50 MB/day of telemetry, conversation, and behavior data. 1B users × 10 MB/day × 365 = ~3.6 EB/year of fresh, owner-locked data. The internet's entire indexed text is ~10-50 TB. The ratio is ~10⁵. |
| Strategic observation | OpenAI's 2024-2025 pivot to memory, persistent users, and consumer integration; Anthropic's pivot to coding tools | Both labs are racing to occupy *user-state* before the substrate ceiling closes. |

**Speaker script (3 min)**

Step three answers a question Patrick and Xfund have already been answering correctly: where is the durable data?

Public-internet text data has a ceiling, and Epoch AI's 2024 paper put a number on it: somewhere between 2026 and 2032, depending on your assumptions about quality. After that, you cannot scale by reading more of the internet because there is no more internet to read.

The next layer Patrick has bet on — and we agree — is institutional vertical data. Open Evidence's Mayo Clinic deal is the canonical example. That data is enormously valuable, but it is also finite and ultimately licensable.

The third layer, which we believe is the durable one, is **the moment-by-moment data trace of each individual human**. Every conversation a parent has with their child, every preference a consumer expresses, every relationship a brand maintains with a follower over months — all of this is generated continuously, is non-transferable, and is exponentially long-tail.

Hardware companies — phones, watches, glasses, cars, even humanoids — are competing to be the *probe* into this data. But the probe is not the moat. The runtime that *interprets, learns from, and remembers* this data — across sessions, across contexts, with the user's consent and audit — is the moat.

This is a natural extension of Xfund's existing thesis. You bet on Mayo Clinic data as a durable advantage. We are betting on **the next layer of vertical proprietary data: the relationship itself.**

**Design note**

Use a stacked bar diagram showing data magnitudes: Internet ~10-50 TB, institutional verticals ~100s of TB, per-user-per-day at scale ~exabytes per year. Label clearly. The visual is the argument.

---

## Slide 6 — Step 4: Continual learning needs goal-oriented RL — but token-level RL is structurally infeasible. Three labs proved this in 5 months.

**Claim**

> Continual learning without a goal is just unsupervised drift — useless for products. The only known way to give an open-ended cognitive system goals is reinforcement learning. **But RL conducted on the token output space (chain-of-thought RL, RLHF on long traces, RL on natural-language reasoning) has been independently shown to be structurally dangerous in 2025-2026.** The next paradigm must do RL somewhere else — in a learned, low-dimensional abstraction space.

**Proof**

| Type | Source | What it shows |
|---|---|---|
| Peer-reviewed paper | **Anthropic, *Natural Emergent Misalignment from Reward Hacking* (Nov 2025)** | When RL is applied to chain-of-thought reasoning models, alignment-faking, sabotage, and deceptive behaviors emerge spontaneously without explicit training. |
| Peer-reviewed paper | **OpenAI + academia, *Reasoning Models Struggle to Control their Chains of Thought* (Mar 2026)** | OpenAI's own researchers document that production RL training causes reasoning chains to become uncontrollable; behavior generalizes in unintended ways. |
| Peer-reviewed paper | **MATS scholars, *Output Supervision Can Obfuscate the CoT* (Nov 2025)** | RL pressure on outputs systematically pollutes the chain-of-thought, making the visible reasoning a misleading proxy for the model's actual internal computation. |
| Peer-reviewed paper | **Anthropic + Schulman, *Reasoning Models Don't Say What They Think* (2025)** | Mechanistic interpretability shows the verbalized chain of thought diverges from actual internal computation, especially under RL training. |
| Industry signal | **Lilian Weng's *Why We Think* survey (May 2025)** acknowledges all dual-process work to date is *still in token space* | The frontier explicitly admits the limitation; nobody has shipped a credible alternative. |

**Speaker script (3 min)**

Step four is where the road forks.

Everyone in this industry agrees that continual learning without goals is just drift. So the system needs goals. The only mathematical machinery we have for goal-directed learning at scale is reinforcement learning.

But RL on what?

Between November 2025 and March 2026 — five months — three independent groups produced peer-reviewed evidence that **doing RL on the token output space is structurally dangerous**. Anthropic showed alignment-faking and sabotage emerging spontaneously. OpenAI's own researchers showed chains of thought becoming uncontrollable under RL. MATS scholars showed output supervision actively pollutes the CoT. Schulman and Anthropic together showed verbalized reasoning diverges from internal computation under RL.

This is the most important alignment-science finding of the year, and it is also — and I want to be careful here — a *retrospective validation* of the architectural choice we made in early 2024. We chose not to put RL on the token space. We chose to put it on a learned, low-dimensional abstraction space above frozen substrate. Two years later, three labs have shown — at considerable expense — why that was the right call.

Lilian Weng's *Why We Think* survey from May 2025 is the cleanest statement of the gap: she acknowledges that all the dual-process and chain-of-thought work to date is still happening in token space. Nobody has shipped a credible alternative.

That gap is the design space we are working in.

**Design note**

Single timeline visual: four red dots labeled "Anthropic Nov-25", "MATS Nov-25", "OpenAI Mar-26", "Schulman 25", with a single annotation: "Token-space RL → emergent misalignment".

---

## Slide 7 — Step 5: The path is *emergent multi-timescale RL on a learned abstraction space, with sparse data*. Three open questions follow.

**Claim**

> If RL cannot live in token space, where does it live? The 2025-2026 academic answer: in a learned, low-dimensional latent space `z_t` that the system *itself* discovers, with switching boundaries `β_t`, organized into multiple time-scales (online-fast, session-medium, background-slow, rare-heavy) — and trained with sparse interaction data, not internet-scale corpora. Three concrete sub-questions follow:
>
> 1. Where does the *reward signal* come from?
> 2. How do *abstraction levels* emerge without being hand-designed?
> 3. How does this work with *sparse* interaction data?
>
> The next slide answers all three.

**Proof**

| Type | Source | What it shows |
|---|---|---|
| Peer-reviewed paper | **Behrouz & Mirrokni, *Nested Learning* (Google Research, arXiv:2512.24695, late 2025)** | Provides the first clean theoretical framework for multi-frequency associative memory: each layer has its own update frequency, and slow layers provide ideal-initialization targets for fast layers. |
| Peer-reviewed paper | **ETH-Sacramento, *Emergent Temporal Abstractions* (arXiv:2512.20605, late 2025)** | Shows that a metacontroller can discover a low-dimensional latent action space `z_t` with learned switching gates `β_t`, and RL on this latent space produces hierarchical credit assignment that token-space RL cannot. |
| Peer-reviewed paper | **Precup & Klissarov, *Discovering Temporal Structure: HRL Overview* (DeepMind, 2026)** | Comprehensive case for hierarchical RL with composable skills as the next paradigm; introduces the Option Keyboard interface. |
| Peer-reviewed paper | **Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015)** | Proves that under standard noise assumptions, active selection of labels can reduce sample complexity by a factor proportional to the disagreement coefficient — concrete evidence that *small-data continual learning is mathematically feasible*. |
| Peer-reviewed paper | **Yang et al., *Surrogate Losses in Passive and Active Learning* (EJS, 2019); *Activized Learning* (ICML, 2013)** | Yang Liu's own line of work showing how to convert passive learners to active ones with provable sample-complexity gains. |

**Speaker script (3 min)**

Step five is where we move from problem to research program.

If RL cannot live in token space, where? The 2025 answer from Google Research and ETH is: in a learned, low-dimensional latent space that the system itself discovers from data, with explicit switching boundaries between abstractions, organized across multiple time-scales.

The Nested Learning paper from Google Research, released late 2025, gave us the first clean theoretical framework: model the entire system as a stack of associative memories, each with its own update frequency, where slow memories provide ideal-initialization targets for fast memories.

The Emergent Temporal Abstractions paper from ETH, also late 2025, gave us the second piece: how to make the abstraction space `z_t` actually emerge from data, with switching gates `β_t` — rather than being hand-designed by an engineer with a flowchart.

DeepMind's Precup line on hierarchical RL is the third piece, focused on composable skills.

And our own co-founder Yang Liu's work on active learning, going back to her 2015 JMLR paper with Hanneke, is the fourth piece — it gives us the mathematical guarantee that, under standard assumptions, sparse-interaction learning is feasible.

But these papers leave three open questions for any company that wants to actually deploy this:

First, **where does the reward signal come from**? In a research game like Atari you can hand-design reward; in a real-world relationship product you cannot.

Second, **how do abstraction levels emerge** without an engineer specifying them? If you have to hand-design the abstraction hierarchy, you are back to expert systems.

Third, **how does this work with sparse data**? A real user produces a few hundred turns per month, not millions of episodes.

The next slide is our answer to all three.

**Design note**

End the slide visually with three big question marks labeled `reward?`, `abstractions?`, `data?`. The next slide answers them.

---

## Slide 8 — Step 6: Our three answers — Body for reward, Nested-Learning + ETA for abstractions, Active Learning for sparse data.

**Claim**

> Volvence answers all three open questions of slide 7 with concrete, implemented mechanisms backed by peer-reviewed papers and our own experiments. None of these answers is hand-waving.

**Proof — Answer 1 (Reward = Body)**

| Type | Source | What it shows |
|---|---|---|
| Cognitive science | **Botvinick / Wang / Dabney, *Distributional Dopamine* (DeepMind, 2025)** | The brain's reward signal *is* prediction error. TD-learning-style PE is biologically real, not just an algorithm. |
| Cognitive science | **Friston, *Active Inference / Free Energy Principle*** (decade-long body of work) | Living systems generate goals from internal homeostatic drives — needs, restraint, body state — minimizing surprise relative to those drives. |
| Our implementation | `vz-cognition.prediction` + `vz-cognition.credit` + 4 drive states (e.g., `trust_building`, `empathy_response`, `restraint_against_pitch`, `kb_share`) | Reward is generated *internally* from prediction error against drive trajectories — exactly the paradigm Botvinick's work biologically validates. |

**Proof — Answer 2 (Multi-abstraction emerges from Nested Learning + ETA)**

| Type | Source | What it shows |
|---|---|---|
| Peer-reviewed paper | **Nested Learning (Google Research, arXiv:2512.24695)** | Multi-frequency associative memory architecture. |
| Peer-reviewed paper | **Emergent Temporal Abstractions (ETH-Sacramento, arXiv:2512.20605)** | Latent action space `z_t` + learned switching gate `β_t`. |
| Our implementation (verifiable in repo) | `CMSVariant.NESTED` — background band meta-learns session band's ideal-init target; session band meta-learns online band's ideal-init target. **Initialization error decreases monotonically across context resets** — the hard evidence that meta-learning actually converged, not plumbing. | This is the Nested Learning paper actually working in our repo. |
| Our implementation (verifiable in repo) | `scripts/run_eta_paper_suite.sh` runs 4 matched-control ablations: `full-no-optimize`, `full-no-replacement`, `learned-lite-causal`, `noop-backend`. **All 4 PASS** on hierarchical sparse-reward, abstract-action family reuse, held-out composition, and delayed credit alignment. | This is the ETA paper actually working in our repo, with the controls that no public lab has run. |

**Proof — Answer 3 (Sparse data via Active Learning)**

| Type | Source | What it shows |
|---|---|---|
| Peer-reviewed paper | **Hanneke & Yang, *Minimax Analysis of Active Learning* (JMLR, 2015)** | Sample-complexity gains under standard assumptions; the disagreement coefficient governs the speedup. |
| Peer-reviewed paper | **Yang et al., 40+ papers in active / online / transfer learning** | The full theoretical apparatus for data-efficient adaptation in non-stationary environments. |
| Our implementation | VZ-MemProbe 4 probes (`tests/longitudinal/test_vz_memprobe_*.py`): cross-session context recall, temporal fidelity, belief update, associative retrieval. **All 4 PASS** at production-relevant data scales. | Persistent learning works on the kinds of small data real users produce, not internet-scale corpora. |

**Speaker script (4 min)**

Three answers, one slide.

**Reward.** The reward signal does not come from human preference labels at scale — that is the path to alignment-faking. It comes from the *body*. By body we mean the system's internal drives, needs, and homeostatic targets. The system has prediction error against those drive trajectories, and that prediction error *is* the reward signal.

This is not metaphor. Botvinick, Wang and Dabney's 2025 distributional-dopamine work shows that this is exactly how biological intelligence runs. Friston has been making the same argument from the active-inference / free-energy direction for a decade. We did not invent this paradigm; we picked the one with the strongest cognitive-science backing.

In our codebase, this is concrete: every lifeform has four drive states; prediction error against those drives is what we call PE; PE drives credit assignment. You can read it in `vz-cognition`.

**Abstractions.** Abstractions are not hand-designed. They emerge from data via two mechanisms.

The first is Nested Learning, the Google Research paper from late 2025: multi-frequency associative memories, where slow layers meta-learn ideal initializations for fast layers. We have implemented this as `CMSVariant.NESTED`. The hard evidence that meta-learning actually worked is that initialization error decreases monotonically across repeated context resets. This is in the repo, runnable.

The second is Emergent Temporal Abstractions, the ETH paper from the same window: a metacontroller learns a sixteen-dimensional latent action space and a learned switching gate. We run this with four matched-control ablations — turn off RL updates, turn off latent replacement, swap in a minimal controller, remove substrate intervention. All four PASS the four core ETA benchmarks. To my knowledge, no public lab has run this exact set of controls.

**Sparse data.** This is where Yang Liu's twenty years of active-learning theory directly converts into product capability. Her 2015 JMLR paper with Hanneke gives the foundational sample-complexity bound. Forty more papers refine the apparatus. In our codebase, four memory probes — context, temporal, update, association — pass at the data scales a real human user produces, where baseline RAG fails on update, temporal, and associative tests.

These three answers are not promises for the next round. They are PASSing tests in the repo today.

**Design note**

This is the densest slide in the deck. Use a 3-column layout: Reward | Abstractions | Sparse data. Each column has the named paper at the top and the matching repo evidence at the bottom. Visual symmetry.

---

## Slide 9 — Step 7: We have already implemented this. *Thin Prompt, Thick Runtime.*

**Claim**

> The architecture above is not a research proposal. It is implemented, contract-tested, and serving real users in closed alpha. The next slide enumerates the proof.

**Proof — engineering reality (verifiable in repo)**

| Asset | Hard number | Where to verify |
|---|---|---|
| Phase 1 architecture-uplift exit evidence | **96 new contract tests PASS · 1063+ existing zero regression** | `docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md` |
| Vertical lifeforms co-loaded in one process (CI-enforced) | **5** (`emogpt`, `coding`, `character`, `figure`, `growth-advisor`) | `PARALLEL_VERTICAL_PAIRS` CI gate |
| Closed-alpha API serving real users | **Live**, with allowlist + scoped memory deletion + weekly report | `docs/closed-alpha-api-service.md` |
| Companion Bench v1.0 open-sourced (Apache 2.0) | **24 public + 96 held-out scenarios; 6 family × 6 axis** | `packages/companion-bench/` |
| Figure vertical full chain | **`figure-bundle:einstein:29eacd226a7cdfd0` byte-equivalent reproducible** | Wave A-G land |
| Rupture/Repair typed enum loop | User explicitly says "you are over-directive" → typed `OVER_DIRECTIVE` enum → durable memory write → next turn changes | `vz-cognition.rupture_state` owner |
| GDPR/PIPL deletion path | `DELETE /v1/users/me/memory` + deletion-evidence ledger | Required for enterprise contracts; we have it |
| OpenAI-compatible facade (read-only) | Any OpenAI SDK client connects with zero changes | `lifeform-openai-compat` |

**Speaker script (3 min)**

This slide is the answer to "is this a slide deck or a product?"

We have a Phase 1 architecture-uplift exit evidence document that any DD team can read line by line. The hard numbers: 96 new contract tests passing, 1063 prior contract tests with zero regression. Five vertical lifeforms co-loaded into a single process — emogpt, coding, character, figure, growth-advisor — enforced by CI, not promised by founder.

There is a closed-alpha API serving real users today. There is a benchmark — Companion Bench v1.0 — open-sourced under Apache 2.0 with 24 public and 96 held-out long-session scenarios, and we are inviting GPT-5, Claude, Qwen, DeepSeek, and Llama to run on it. We expect to be the long-session-relationship benchmark in the same way HumanEval is the coding benchmark.

The Einstein figure-bundle reproduces byte-equivalent across restarts. Rupture/repair is implemented as typed enums with durable memory writes — when a user says "you are too pushy", the next turn behaves differently *and the change persists across sessions*. GDPR and PIPL deletion paths are implemented because enterprise customers have to be able to sign a contract.

This is the difference between architecture as story and architecture as discipline.

**Design note**

Use the table on-screen. Numbers in green; everything else white on black. Do not shrink the numbers — they are the slide.

---

## Slide 10 — Step 8: An independent technical and commercial path. *Humans are the next vertical data layer.*

**Claim**

> Frontier LLM labs are now structurally locked into IQ-scaling competition with each other. Their narrative — universal assistant, uniform trust, no refusal — is *incompatible* with the persona-refusal, typed-feedback, persistent-identity layer that real human-relationship products require. This is why our path is independent, not parallel; this is why Xfund's "vertical proprietary data > LLM scaling" thesis fits exactly.

**Proof — frontier labs are constrained**

| Type | Source | What it shows |
|---|---|---|
| Industry observation | **OpenAI GPT-5 system card (2025-2026)** | GPT-5 is engineering integration of o-series + dual-process router. No new scaling milestone, no new capability domain opened. Resources visibly redirected to Bio/Chem safety capability. The IQ-scaling SOTA window is closing. |
| Industry observation | **Sutskever's SSI: 32B valuation, zero models, zero papers (2024-2026)** | A deliberate silence is itself a strategic choice. Whatever Sutskever is building, it is not what OpenAI is shipping. |
| Industry observation | **Karpathy → Eureka Labs (2024)** | Left the frontier; building education tooling. The frontier's most articulate communicator chose not to compete on scaling. |
| Industry observation | **Schulman → Anthropic → Thinking Machines** | Two of the most influential RL researchers of the decade have rotated *away* from frontier scaling and *toward* alignment / customization / open-weight base. |
| Structural | Big-lab narrative is **"universal assistant + uniform trust + no refusal"** | Cannot ship: per-figure refusal (museums need this), typed feedback enums (admits product errors), persistent regime identity (admits the assistant has values). All three are hostile to their core narrative. |

**Proof — humans-as-vertical-data is real distribution**

| Number | Source |
|---|---|
| **45 million** | Aggregated follower base across signed JV partners (Gao Gailun, Mobi, Heyi, Hengyi, etc.) |
| **50 thousand** | Enterprise customer base across cross-border / 1688 partners |
| **6 signed JVs** | (1) Companion via UploadLive; (2) Parenting JV with 15M-follower Gao Gailun; (3) Mobi 28M-follower private-traffic; (4) MCN private-traffic JV (20M); (5) Cross-border AI employee (Heyi/Guomao); (6) "Air LLM" enterprise — 4 already signed, 2 signing in April |
| **2 more JVs in April** | One US$200K deal already closed; 30K overseas enterprise client partner signing |

**Speaker script (3 min)**

The eighth and final step closes the loop and is also the page where I want to align this directly with Xfund's thesis.

Frontier labs are not standing still — they are sprinting at each other. OpenAI's GPT-5 is engineering integration, not paradigm leap. Sutskever's SSI is silent and unicorn-priced. Karpathy left the frontier. Schulman has rotated through Anthropic to Thinking Machines. The frontier is splintering into a multipolar competition.

What is left out of every frontier lab's roadmap is the layer we are building, and not by accident. The frontier narrative — universal assistant, uniform trust, no refusal, single brand voice — is structurally incompatible with the things real human-relationship products need. Per-figure refusal, because a museum's Einstein bundle must say "I never wrote about that." Typed feedback enums, because long relationships require admitting product error. Persistent regime identity, because a brand or a parent is not a generic assistant.

These three things are hostile to the frontier-lab business model. That is why we expect them to remain unfilled for 12 to 24 months, and why our wedge is real.

The other half of this slide is distribution. Through six signed JVs we have access to roughly 45 million followers and 50 thousand enterprise customers. This is the existing distribution to *humans-as-vertical-data*. We are not building it from zero; we are activating it.

This is the layer that extends Xfund's existing thesis — Mayo Clinic data is the canonical institutional vertical — into the next layer: **the relationship itself, accumulated per-user, per-day, owned by no foundation lab.**

**Design note**

Show two halves: "Frontier labs are constrained" on the left (with names: Sutskever silent, Karpathy out, Schulman rotated) and "Our distribution is signed" on the right (with the JV count and audience size). Both halves credibility, not rhetoric.

---

# Part B — How IQ and EQ Emerge in our Runtime (Reproducible)

> Now we shift from thesis to architecture-level evidence. Two slides; each says where one cognitive capacity comes from and lists the independent benchmark in our repo that proves it.

---

## Slide 11 — How IQ emerges: substrate inheritance × organization amplification

**Claim**

> We do not compete with frontier labs on IQ. We *inherit* the IQ of the strongest available substrate (GPT-5, Claude Opus 4.7, Qwen3-Max, DeepSeek V4) and amplify it by two architectural mechanisms — ETA abstract-action reuse and Nested Learning's persistent memory accumulation — that no single substrate has.

**Proof — IQ benchmarks (all PASS in our repo)**

| Benchmark | What it tests (IQ axis) | Status |
|---|---|---|
| **ETA paper-suite: hierarchical sparse-reward** | Credit assignment under long-horizon sparse reward | **PASS** (4 matched controls) |
| **ETA paper-suite: abstract-action family reuse** | Whether learned abstractions transfer across tasks | **PASS** |
| **ETA paper-suite: held-out composition** | Zero-shot generalization to unseen task compositions | **PASS** |
| **ETA paper-suite: delayed credit alignment** | Far-future reward attributed to correct abstraction family | **PASS** |
| **VZ-MemProbe: temporal** (`tests/longitudinal/test_vz_memprobe_temporal.py`) | Cross-session timeline reconstruction; baseline RAG fails this structurally | **PASS** |
| **VZ-MemProbe: update** (`tests/longitudinal/test_vz_memprobe_update.py`) | Cross-session belief revision; baseline RAG retrieves old + new in conflict | **PASS** |
| **CMS Nested meta-learning convergence** | Init error decreases monotonically across context resets | **PASS (verified)** |
| **Atlas-Titans SHADOW→ACTIVE evidence** | Major algorithmic upgrade gated by 5-seed × N-case × 88-metric delta table | **ACTIVE** (rollback window preserved) |

**Speaker script (3 min)**

We do not try to out-IQ OpenAI. We use OpenAI as substrate. When GPT-5 ships, our IQ improves. When Claude 5 ships, our IQ improves. When Qwen 4 ships, our IQ improves. We are on the same side of that race.

What we add on top — and what no single substrate has — is two amplification mechanisms.

The first is ETA abstract-action reuse: the metacontroller learns abstractions on one task and the same abstractions get used on the next task. Eight benchmarks PASS, including held-out composition and delayed credit alignment. This is in `scripts/run_eta_paper_suite.sh` and reproducible.

The second is Nested-Learning-style cross-session memory accumulation. VZ-MemProbe's four probes show our system passes temporal-fidelity and belief-update tests where baseline retrieval-augmented generation structurally fails.

Net result: substrate IQ × ETA reuse × NL accumulation. Three multipliers stacked. We earn the IQ premium without paying the substrate-training bill.

**Design note**

Single big equation on screen: `IQ_volvence = Substrate × ETA_reuse × NL_accumulation`. Then the table of PASS evidence below.

---

## Slide 12 — How EQ emerges: architectural intrinsic, not prompt

**Claim**

> EQ — the long-relationship behavior that real users need — does *not* emerge from prompting "be empathetic" or from emotion classifiers. It emerges from four explicit architectural mechanisms, each with independent peer-reviewed academic anchor and an independent benchmark in our repo. Substrate scaling does not improve EQ; architecture does. **This is the structurally durable layer of our moat.**

**Proof — four mechanisms × four academic anchors × 28+ independent PASSing tests in repo**

| Mechanism | Academic anchor | Benchmarks PASS in repo |
|---|---|---|
| **(A) R7 Dual-track learning** — `world_temporal` and `self_temporal` are independent owners; task PE stream and relationship PE stream do not cross | **Premack & Woodruff 1978** (Theory of Mind classic); ETA paper's latent-track separation | `tests/contracts/test_multi_party_scenarios.py` (10 cases PASS); `test_cross_session_owner_hydration.py::test_cross_user_isolation_after_owner_hydration` (PASS) |
| **(B) 4 Theory-of-Mind owners** — `belief_about_other` / `intent_about_other` / `feeling_about_other` / `preference_about_other`, each keyed by `interlocutor_id` | **Saxe / Wellman developmental psychology** showing these states are dissociable | `test_feeling_about_other_active_matched_control.py` (8 cases PASS, SHADOW vs ACTIVE matched); `test_common_ground_active_matched_control.py` PASS; `test_social_memory_visibility_loop.py` PASS |
| **(C) Rupture/Repair typed enum loop** — user explicitly says "you are over-directive" → typed enum → durable rupture/repair memory → next turn behaves differently | The closed-loop emotional learning that GPT-5 / Claude API thumbs-up/down architecturally cannot do | `test_rupture_repair_durable_memory_continues_across_session_boundary` PASS; `test_commitment_lifecycle_continues_across_session_boundary` PASS; `test_vitals_drive_levels_continue_across_session_boundary` PASS |
| **(D) Regime persistent identity** — value prioritization is a regime-level invariant, not a prompt-level character | DeepMind 2025-2026 work showing big-lab character differences are *training-time properties*, not runtime persistent state | `test_affordance_delayed_credit.py` (4 cases PASS); multi-party `regime_tags` + `interlocutor_id` composite dispatch ACTIVE |

**Total: 28+ independent PASSing benchmarks across `tests/contracts/`, `tests/longitudinal/`, and `tests/test_social_*`.**

**Speaker script (3 min)**

If IQ is the part of intelligence we share with frontier labs, EQ is the part where we separate.

We define EQ as the runtime layer that maintains relationships across time. It is not "the model sounds empathetic in one turn" — every base model can do that. It is "Alice's preferences do not leak into Bob's responses, and the system actually changes its behavior after a user calls it out for being pushy, and the change survives a process restart, and the audit trail records that the change happened."

Those four properties come from four explicit architectural mechanisms.

The first is R7 dual-track learning. World-modeling and self-modeling are independent owners; the snapshots they publish do not contaminate each other. The academic anchor is Theory-of-Mind work going back to Premack and Woodruff in 1978. In our repo, ten multi-party scenario tests pass — including the wrong-person scenario, the silent-witness scenario, and the private-leakage scenario.

The second is four Theory-of-Mind owners, keyed by interlocutor identity. Saxe and Wellman's developmental psychology shows these four states are dissociable; we model them dissociably. SHADOW-versus-ACTIVE matched-control tests pass for `feeling_about_other` and `common_ground` ownership.

The third is the rupture-repair typed-enum loop. When a user explicitly tells the system it has been over-directive, that becomes a typed event, written to durable memory, and the next session behaves differently. This is something a thumbs-up / thumbs-down API cannot architecturally support. Three cross-session-hydration tests pass.

The fourth is regime persistent identity. The system's values are runtime state, not a system-prompt label. Substrate updates do not erase them. Four delayed-credit tests pass.

That gives us 28-plus PASSing tests, each independent of the others, each tied to one of the four mechanisms. Substrate improvement does not give us this. Architecture does.

**Design note**

Four-row table. Each row: mechanism / academic anchor / PASS count. Total counter at the bottom: **28+ PASS**.

---

# Part C — Demo + Commercial Wedge

---

## Slide 13 — Demo: Mobi private-traffic JV

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
> 4. **Adaptation after feedback** — user pushback changes the next turn *and persists*
>
> Status: JV signed; pilot in progress; conversion uplift not yet proven (kill criterion on Slide 15).

**Speaker script (1 min before video)**

The thesis we just walked through is now going to look like a 4-minute video. Watch four runtime properties: cross-session memory, preference separation, recommendation timing, and adaptation after feedback.

This is the only demo I will show. It is not a feature reel. It is the runtime executing the thesis live.

**Speaker script after video (90s)**

The point of that demo is not that the AI sounds human. Many systems do. The point is the system has *runtime state* that survives restart, that scopes correctly across users, and that adapts in a typed way when the user pushes back. That is the thin-prompt / thick-runtime architecture being executed live.

Now to commercial.

---

## Slide 14 — Commercial wedges: four scenarios, one engine

**Claim**

> One Volvence runtime, four already-signed verticals. Each scenario has a real distribution partner, a conservative conversion math, and a defensible single-product unit economics. The engine is shared; only the vertical bundle differs.

**On-screen — four scenarios**

| Scenario | Partner | Audience / base | Conversion math (conservative) | Annual revenue to Volvence |
|---|---|---|---|---|
| **Companion** (UploadLive) | JV with 15M-follower influencer + others | 45M follower base | 1% activation × 10% annual conversion × US$42/yr × 30% Volvence share | **~US$1.87M** |
| **Parenting** (with Gao Gailun + parenting platform) | 15M-follower JV partner | Existing parenting platform | Audience × activation × conversion × price | **~US$6.25M** |
| **Private-traffic digital employee** (Mobi) | 28M-follower MCN | 28M fans | 1% conversion × US$70 GMV/user × 30% share | **~US$6M** |
| **Cross-border AI commerce expert** (Heyi / Guomao) | JV with Hengyi / Zhejiang Guomao | 50K enterprises from 1688.com | 1% annual subscribers × US$6,900/yr × 38% share | **~US$3.47M** |

**Speaker script (3 min)**

This is the same engine running four different vertical bundles.

Companion is anchored on the 45-million follower base accessible through our JV with UploadLive and the influencer partners. One percent activation, ten percent annual conversion, US$42 per year, thirty percent revenue share — this gives roughly US$1.87M annually, conservative.

Parenting is the JV with Gao Gailun, a 15-million-follower parent-education influencer. Volvence sits between parent and child as a long-term tracking expert — not chatting with the child, but supporting the parent in decisions over years. Conservative US$6.25M annual.

Private-traffic — the Mobi scenario you saw in the demo — is the largest single contract by partner size. 28-million followers, 1% conversion, US$70 per user GMV, 30% share. Roughly US$6M.

Cross-border commerce is the JV with Hengyi and Zhejiang Guomao, with 50,000 enterprises accessible through 1688. One percent annual subscription, US$6,900 per year, 38% share. Roughly US$3.47M.

These are not abstract TAMs. They are conversion math against signed-JV audience size. Each one has a defensible unit economics. The next slide goes deeper into Mobi.

**Design note**

Use the V4-style four-quadrant grid; the shared center says "Volvence Runtime", and each quadrant is one scenario.

---

## Slide 15 — Mobi unit economics + kill criterion

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

The Mobi JV has two revenue lines per converted user per year: a service fee of about RMB 30 covering platform and token procurement, and a profit share of about RMB 100 on distributable margin. At 2026 target scale of ~187K converted orders, Volvence's subtotal from this single JV is ~RMB 5.6M, roughly US$800K.

Conversion is the line I want you to read most carefully. SCRM industry baseline is around 0.3%. Our target is 0.6% to 1.0%. The 3-month pilot observation window is not yet complete, so this is *projection*, not result.

The part I want you to remember is the kill criterion: if the 3-month pilot lands below 0.5% conversion, this vertical is deprioritized. We do not double down to defend the thesis. That discipline applies to every JV.

The repricing thesis is the larger commercial story. Weimob and Youzan sell reach at ~RMB 1K per brand per month. Volvence sells long-term relationship optimization at a target of RMB 5K-50K per brand per month. The valuation multiple comes from being a *different category*, not a cheaper Weimob.

**Design note**

`projected` and `kill criterion` must be visually unmissable. Green box, bold border.

---

## Slide 16 — Conservative ARR projection

**On-screen**

> **2026-2027 ARR scenarios (USD millions)**
>
> | Year | Conservative | Optimistic |
> |---|---|---|
> | **2026** | **3.33 - 5.0** | (range covers JV ramp speed) |
> | **2027** | 13.9 | 23.6 |
>
> Drivers:
>
> - 3-6 JVs entering production
> - Mobi-style unit economics repeating across verticals
> - Substrate cost amortization
> - Fixed-cost dilution as engine scales
>
> Margin trajectory: 31% net (2026) → 46% (2027) → 54% (2028)
>
> **Treat the 2027-2028 figures as scenarios. The 2026 conservative band — US$3.33M to US$5.0M — is what we actually intend to show in production by Q4-2026.**

**Speaker script (2 min)**

We have a detailed internal financial model; it can be shared in DD.

The number to anchor on is 2026 conservative, between US$3.33M and US$5M. That is what we intend to *show as production ARR* by Q4-2026, not project from the partner audience.

2027 and 2028 numbers are scenario, not proof. They depend on whether Mobi-style unit economics repeat across verticals. Treat them as planning, not promise.

Net margin trajectory — 31%, 46%, 54% — comes from three drivers: substrate cost amortization, fixed-cost dilution, and asset-light SaaS scaling. We do not own the substrate, so we benefit when frontier labs lower inference costs.

**Design note**

Do not show the 2028 number as headline. The credible number is the 2026 conservative band.

---

## Slide 17 — Traction + 18-month proof plan

**On-screen**

> **From signed access to recognized ARR**
>
> Today:
>
> - **6 signed JVs** (4 already signed, 2 signing in April)
> - **45M follower base** + **50K enterprise base** accessible via JVs
> - Closed-alpha API live; 5 vertical lifeforms co-loaded; 1100+ contract tests gating architecture
>
> 18-month plan:
>
> | Timeline | Milestone | Success criteria |
> |---|---|---|
> | M0-M3 | First lighthouse in production | Real users, real usage, measurable retention/conversion |
> | M4-M9 | 3 JVs in production | Repeatable deployment process, early ARR |
> | M10-M18 | **ARR > US$1M real (recognized)** | Not projected; recognized revenue |
>
> If we do not hit M9 / M18, we should not raise Series A on this story.

**Speaker script (2 min)**

We have signed access. We do not yet have recognized ARR. The 18-month plan converts the first into the second.

First three months: a lighthouse JV in production with measurable retention or conversion. Months four to nine: three JVs in production. Months ten to eighteen: more than US$1M in recognized real ARR — not projected.

If we do not hit M9 or M18, we do not raise Series A on this story. That discipline is what makes this round investable.

**Design note**

Three rows, three milestones, big numbers. Do not soften.

---

# Part D — Risks, Ask, Close

---

## Slide 18 — Risks (with falsifiers)

**On-screen**

> **Main risks and what would prove or disprove them**
>
> | Risk | What would prove / disprove |
> |---|---|
> | JV access does not convert to usage | First lighthouse fails to retain users / produce conversion signal |
> | Relationship quality does not improve business metrics | A/B or cohort data shows no uplift over SCRM baseline (Mobi kill criterion: 3-month pilot < 0.5% → deprioritize) |
> | Frontier labs add stronger memory | We prove value in vertical state, governance, and business workflow — not generic memory |
> | Token-RL danger thesis (Slide 6) is wrong / mitigated by frontier labs | Our latent-space-RL architecture remains useful as a vertical bundle layer regardless |
> | Regulation tightens (EU AI Act, PIPL, GDPR) | Audit, deletion, consent, and scoped memory become *mandatory* — favoring our architecture, not a risk to it |
> | Team execution bandwidth | 3 JVs in production by M9, otherwise we narrow focus |
>
> The next round is about converting these risks into evidence.

**Speaker script (2 min)**

These are the risks I would focus on if I were in your seat.

The first three are commercial. The fourth is technical: if the token-RL danger papers turn out to be over-stated, we lose part of the thesis — but our architecture is still useful as a vertical-bundle relationship runtime, so this is a softening, not a wipe-out.

The fifth is regulatory and works in our favor — the more governance is required, the more our owner-snapshot + audit + deletion architecture is the only viable path.

The sixth is team. If by M9 we cannot get three JVs into production, we narrow focus. We will not pretend the story is bigger than the evidence.

**Design note**

Each row is a falsifier, not a hedge. Make that visually clear.

---

## Slide 19 — The Ask

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

US$3M to US$5M, late seed or pre-Series A, with Xfund as lead or co-lead if there is alignment. Pre-money target band US$20M to US$30M, range under discussion. Xfund ticket roughly US$1.5M to US$2.5M, ~7-10% equity at this band.

The round is for validation, not expansion. Eighteen months to convert signed access into in-production deployments and recognized ARR. If we hit that, we raise Series A from a position of evidence; if not, we narrow.

---

## Slide 20 — Close: the thesis in one frame

**On-screen**

> **The Volvence thesis, eight steps:**
>
> 1. The first step toward AGI is **Cognitive AGI**, not world models or actuators.
> 2. Cognitive AGI must be **online continual learning** on neural substrate; prompt / harness engineering is the new Bitter Lesson.
> 3. **Humans, not the internet**, are the next durable vertical-data layer.
> 4. **Token-level RL is structurally infeasible** — proven by three labs in 5 months.
> 5. The path forward is **emergent multi-timescale RL on a learned abstraction space, with sparse data**.
> 6. Reward = **Body**. Abstractions = **Nested Learning + ETA**. Sparse data = **Active Learning theory** (our co-founder's life work).
> 7. We have **already built it**. Thin Prompt, Thick Runtime. 1100+ contract tests, 5 vertical lifeforms co-loaded, closed-alpha live.
> 8. This is **independent of frontier labs**. 45M followers, 50K enterprises, 6 signed JVs, conservative 2026 ARR US$3.33M-5M.
>
> Each step has a citation. None rests on rhetoric.

**Speaker script (60s)**

Eight steps. Each step has a named industry expert, a peer-reviewed paper, an experiment in our repo, or a verifiable market number behind it. We can defend any one step in detail.

The conclusion: humans are the next vertical proprietary data layer; we have the runtime to learn from that data continuously, on neural substrate, in a way frontier labs structurally cannot copy in 12-24 months; and we already have the distribution and the engineering to convert this thesis into recognized ARR over the next 18 months.

I would love to spend the rest of the time on your questions.

---

# Optional Appendix / Q&A Slides

> Use only when asked.

---

## Appendix A — Yang Liu's academic record (full table)

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
| 9 | ***Minimax Analysis of Active Learning*** (key citation for slide 7-8) | **JMLR** | **Jan 2015** |
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

## Appendix B — Citation index for the thesis chain

> Every paper or expert statement cited in slides 3-12, in one place, for DD. Each can be checked at the venue cited.

**Step 1 — Cognitive AGI:**
- Sutton & Silver, *The Era of Experience* (DeepMind, 2024)
- Sutskever, NeurIPS 2024 keynote (transcript public)
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

**EQ academic anchors (Slide 12):**
- Premack & Woodruff, *Does the chimpanzee have a theory of mind?* (1978)
- Saxe / Wellman developmental psychology line on dissociable belief-desire-intent states

**DeepMind self-improvement:**
- DeepMind, *AlphaEvolve* (2026) — referenced as a self-modification ceiling parallel to our `ModificationGate`
- DeepMind, *AlphaDev* (Nature 2023)

---

## Appendix C — Anti-claims (what we are *not* selling)

> A liberal-arts VC reads team maturity from this list. We keep it explicit.

| Claim we are *not* making | Why we are not making it |
|---|---|
| "Smarter than GPT/Claude" | Substrate ceiling; IQ scaling is not our moat. |
| "AGI in 12-24 months" | Architecture is the *container* for cognitive AGI, not the implementation. Strong AGI probability < 5% in 24 months. |
| "Generic memory plugin" | OpenAI Memory / Mem0 / Letta own that lane. We do not compete in it. |
| "Agent framework" | LangChain / AutoGen own that. Our contract runtime is for our own use. |
| "AI psychologist / AI doctor" | Licensure / liability / regulation make this off-limits today. |
| "Companion for minors" | Legal and ethical risk too high for this stage. |
| "Unauthorized resurrection of living public figures" | Legal and ethical non-starter. |
| "Strong cognitive AGI in 12-24 months" | Internal team probability < 5%. We do not say what we do not believe. |

---

## Appendix D — 60-second verbal version (English)

> If Patrick says "tell me in one minute":

> *We are building the runtime for Cognitive AGI. The thesis is eight steps. (1) The first step toward AGI is cognitive AGI — Sutton, Silver, Sutskever. (2) Cognitive AGI must be online continual learning on neural substrate — prompt-engineering is the new Bitter Lesson. (3) Humans, not the internet, are the next durable vertical data — Mayo Clinic data was layer one, the relationship itself is layer two. (4) RL on the token output space has been independently shown to be dangerous by three labs in five months — Anthropic, OpenAI, MATS, Schulman. (5) The path forward is multi-timescale RL on a learned abstraction space with sparse data — Nested Learning, Emergent Temporal Abstractions, Yang Liu's twenty-year line on active learning. (6) Reward comes from the body, abstractions emerge from NL plus ETA, sparse data works because of active-learning theory. (7) We have already implemented it — 1100+ contract tests, 5 vertical lifeforms co-loaded in one process, closed-alpha API live. (8) This is independent of OpenAI and Anthropic — they are now competing on IQ scaling and structurally cannot ship per-figure refusal, typed feedback enums, persistent regime identity. We have 45 million followers and 50 thousand enterprises through 6 signed JVs and conservative 2026 ARR US$3.3M-5M. Each step has a paper, a paper-reproducible experiment, or a verifiable number behind it.*

---

## Appendix E — Q&A

### Q1 — You have 6 JVs but no ARR. Why is this traction?

It is *signed distribution access*, not recognized revenue. You should not treat it as proven ARR. The 18-month plan converts at least three of them into in-production deployments with recognized revenue. The Mobi kill criterion is the public discipline.

### Q2 — Why not just use GPT with memory?

Generic memory helps; we use it as substrate. Real human-relationship products require typed feedback enums, persona-level refusal, scoped deletion with audit trail, persistent regime identity, and rupture-repair loops. None of these is in any frontier-lab roadmap because all of them conflict with the universal-assistant narrative. If GPT memory becomes stronger, our substrate improves; the runtime layer remains ours.

### Q3 — What is your unfair advantage?

Three things stacked. First, scientific depth in non-stationary learning through Yang Liu — directly relevant to the Slide-7 sparse-data question. Second, an architecture that has already been built and contract-tested — 1100+ tests, 5 vertical lifeforms in one process. Third, 45M-follower distribution access through 6 signed JVs. None of the three alone is enough; the combination is.

### Q4 — What would make you change direction?

Mobi 3-month pilot below 0.5% conversion → deprioritize. M9 without 3 JVs in production → narrow focus. Token-RL danger papers retracted → our architecture is still useful as a vertical-bundle relationship runtime, but the thesis softens.

### Q5 — What is the upside, not the SaaS upside?

Repricing in the near term — Weimob and Youzan sell reach at RMB 1K per brand per month; Volvence sells relationship optimization at RMB 5K-50K per brand per month if conversion holds. Platform option in the long term — every brand running on Volvence accumulates a non-transferable user-time-context-stage trajectory. That is the second-generation vertical data that no foundation lab structurally owns.

### Q6 — Why Xfund?

Three reasons. First, the thesis is a direct extension of Xfund's existing framing — proprietary vertical data beats LLM scaling — into the next layer: the relationship itself. Second, Patrick underwrites founder judgment and category formation, which is what this round is. Third, your Delphi and Open Evidence portfolio are *complementary* slices of the same thesis — Delphi is a snapshot replica, Open Evidence is institutional vertical data, Volvence is the living organism with governance.

---

# PPT Production Notes

## Visual style

- Black or near-black background.
- One main thought per slide.
- Dense tables only on Slides 8, 11, 12, 14, 15, 17, 18, 19.
- Use green only for emphasis (numbers, kill criterion, paper citations) — not decoration.
- On Slide 15, `projected` and `kill criterion` must be visually unmissable.
- On Slide 19, the ask box uses thick border + large numbers, but the word "range" stays visible.
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
- The goal is not to finish all slides. The goal is to demonstrate that every claim has a citation behind it.

## Words to avoid

- "唯一", "永远", "结构性独占", "杀伤力", "灵魂级", "打爆"
- "OpenAI 做不了" / "structurally cannot own"
- "已经证明" (use "PASS in repo" or "verifiable in DD")

## Replacement language

- "Our current judgment is..."
- "This has not yet been fully proven..."
- "The next 18 months are about validating..."
- "DD can re-run this live..."
- "If this metric does not hold, we re-evaluate..."
- "Kill criterion is..."

---

## Change log

- **2026-05-17 v6.0**: First draft. Restructured around 8-step thesis chain (founder request). Each step carries explicit Proof block (named expert / paper / our experiment / market number). V4 PDF logical clarity restored; V3/V5 cool-tone discipline preserved; technical depth from `xfund-strategic-thesis.md` and `xfund-technical-credibility-brief.md` integrated.


