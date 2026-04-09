# Next-Generation EmoGPT — System Design

> Status: v2 draft
> Scope: system-level thesis and invariants (not implementation spec)
> Sources: Nested Learning (NL, arXiv:2512.24695), Emergent Temporal Abstractions (ETA, arXiv:2512.20605)
> Downstream: `docs/specs/*.md`, `docs/DATA_CONTRACT.md`, `.cursor/rules/`

## Part 0. Purpose and Scope

This document defines the **target-state design invariants** for a next-generation EmoGPT system. It is:

- the single source of truth for **why** the system is built this way
- the grounding for all `docs/specs/*.md` capability specs
- the bridge between NL/ETA academic ideas and our product/engineering decisions

It is **not** an implementation plan, a module API reference, or a paper restatement.

---

## Part 1. Core Thesis

### What the system is

EmoGPT is a **bounded, continuously adapting digital organism** whose core product is relationship and subjectivity (EQ + trust), not only intelligence (IQ). The system is not a static model plus prompts. It is an explicitly layered adaptive system with fast, medium, and slow learning loops.

### Five pillars

1. **Prediction error is the primitive learning signal.**
   The system learns by predicting outcomes and updating itself from the mismatch between prediction and reality. Reward and punishment are not externally labeled — they are derived from prediction error. Following NL: *"Memory is a neural update caused by an input, and learning is the process for acquiring effective and useful memory"* (NL §3.1). The gradient of the loss with respect to the model's output is the **local surprise signal (LSS)** — the fundamental measure of how surprising a prediction was. All credit, evaluation, and reward are downstream of this primitive.

2. **The system is a nested, multi-timescale associative memory system.**
   Following NL's NSAM framework, the entire system — substrate, controllers, optimizers, memory — is modeled as a set of interconnected associative memories at different update frequencies. Architecture and optimizer are not independent design knobs; they are jointly a **neural learning module** whose behavior depends on the whole.

3. **Temporal abstraction is explicit and lives above token generation.**
   Following ETA, the system maintains a formal layer of temporally-abstract internal actions (controller codes `z_t`) above the raw token level, with learned switching conditions (`β_t`). Internal control operates in this compressed latent space, not in the high-dimensional token space. This enables efficient exploration, sparse-reward learning, and meaningful credit assignment.

4. **Relationship and task are distinct prediction spaces.**
   The system tracks two partially separated learning tracks: world/task (predicting task outcomes) and self/relationship (predicting relational consequences). These are not style variants; they are separate prediction error streams with distinct memory, credit, and controller update paths.

5. **The LLM is the expression layer, not the primary learner.**
   The pretrained language model provides the substrate and the voice. Learning, adaptation, and control happen in the layers above it — controllers, memory, reflection, regime — not through prompt engineering or token-level fine-tuning.

---

## Part 2. Design Laws (R1–R15)

Requirements are organized by dependency, not by number. R-IDs are stable and referenced across all specs.

### Foundation: contracts and ownership

**R8. Snapshot-First, Contract-First Architecture**

Every meaningful runtime area has a single primary owner. Cross-module exchange happens through immutable public snapshots. Consumers do not reconstruct producer internals. Enriched publishers are preferred over downstream rebuilding. Runtime controllers may consume substrate state, but must not silently become second owners.

**R15. Migration Must Preserve Explainability and Rollback**

The system evolves by bounded incremental packets. Each adaptive layer has a clear owner. Each public exchange is inspectable. Old and new learning paths have named exit conditions. Rollout is reversible. Evaluation evidence must precede scope widening.

### Identity: dual tracks and being

**R7. Self/Relationship Learning Is Separate from Task Learning**

The architecture tracks two partially separated learning tracks:

- **World/task track**: facts, plans, user situations, external goals — prediction errors about task outcomes
- **Self/relationship track**: trust, attachment, interaction regime, repair history — prediction errors about relational consequences

These tracks share infrastructure but remain semantically distinct in memory writes, credit assignment, controller updates, and evaluation metrics. Relationship continuity is not a side effect of problem solving.

**R12. Evaluation Must Cover Being, Not Only Task Success**

Required evaluation families: task capability, interaction quality, relationship continuity, learning quality, abstraction quality, safety and boundedness. A system that only scores well on one-turn helpfulness is insufficient.

**R14. Social and Cognitive Regimes Need Persistent Identity**

Regimes (casual social, acquaintance building, emotional support, guided exploration, problem solving, repair and de-escalation) are represented in runtime state, recallable from memory, selectable by higher-level control, and trainable through delayed outcomes. They are not prompt labels.

### Learning: prediction error, timescales, and compression

**R-PE. Prediction Error Is the Primitive Learning Signal (new)**

The system must explicitly produce predictions about outcomes before acting and compare them to actual outcomes after the next turn. The difference — prediction error / local surprise signal — is the raw material from which all credit, memory updates, and policy changes are derived. This is not an optional diagnostic; it is the primary signal driving all adaptation.

- Evaluation scores are *readouts* of prediction error, not the source of learning
- Credit records are *aggregations* of prediction error over time and across levels
- Reward and punishment are the sign and magnitude of prediction error

**R1. Multi-Timescale Learning Is Mandatory**

The system operates at explicitly different update frequencies:

- `online-fast`: per turn or per wave adaptation
- `session-medium`: per scene or per conversation adaptation
- `background-slow`: post-session reflection and consolidation
- `rare-heavy`: offline retraining, distillation, or policy refresh

Not all knowledge should live in one parameter block. Not all state should update with the same cadence. Fast adaptation should not require rewriting the whole model. Slow consolidation should not block the live interaction loop.

**R2. Stable Substrate + Adaptive Controllers**

The system distinguishes a relatively stable foundation substrate from higher-level adaptive controllers. The default stance: freeze or slowly update the foundation model; place most online adaptation in bounded controller layers, memory writes, routing policies, and reflection-driven updates.

ETA's rate-distortion analysis demonstrates that freezing the base model is necessary for discovering temporal abstractions — joint training leads to degenerate solutions. NL's frequency-ordered levels reinforce this: different levels must keep clear update boundaries instead of collapsing all learning into one gradient flow.

**R13. The Training Loop Must Alternate Compression and Reinforcement**

Following ETA's wake-sleep cycle and NL's nested levels:

- **SSL phase**: compress interaction history into structured internal representations
- **RL phase**: reinforce controllers and strategies in the compressed representation space

This alternation operates at multiple scales. The invariant: reinforcement should act on a compressed and structured internal substrate, not on raw behavior alone.

### Control: temporal abstraction and internal RL

**R3. Temporal Abstraction Is a First-Class Capability**

The system supports a formal layer of temporally-abstract actions above token generation. Following ETA:

- The metacontroller produces controller codes `z_t` and switching gates `β_t`
- `β_t ≈ 0`: persist current abstract action; `β_t ≈ 1`: switch to new action
- The decoder maps `z_t` to residual stream controller parameters `U_t`
- Each abstract action executes a sequence of behaviorally meaningful steps
- Switching patterns emerge from the variational bottleneck (`α · D_KL`), not from explicit labels

Product mapping: abstract actions correspond to strategies like trust repair, guided exploration, listen-first mode, or collaborative planning.

**R4. Internal Control Happens Above Raw Token Space**

Following ETA's Internal RL:

- Stage 1 (SSL): train a non-causal encoder to discover `z_t` and `β_t` via variational objective (Eq.3)
- Stage 2 (Internal RL): replace the non-causal encoder with a causal policy `π(z_t | e_{1:t})`, freeze everything else, and train with RL

The action space is `z_t` (low-dimensional latent codes), not raw tokens. The environment includes the frozen autoregressive model plus the metacontroller's decoder and switch unit. Advantages: action space dimensionality reduction (`n_z ≪ n_e`), temporal compression (abstract action timescale), simplified credit assignment (sparse switching), and exploration efficiency (sampling `z ~ N(0,I)` produces meaningful behavior sequences).

### Memory: continuum and consolidation

**R5. Memory Is a Continuum, Not a Binary Split**

Required strata: transient working state, session episodic state, durable semantic memory, derived indexes. Each stratum has different update frequencies, promotion/decay rules, and reconstruction capabilities.

**Paradigm-level understanding** (NL): memory is any input-driven neural update distributed across all parameters. Learning is acquiring useful memory. This includes CMS bands, controller weights, optimizer state, and anything that changes in response to input.

**Runtime-level implementation**: the system uses explicit owner modules (MemoryStore, CMS, snapshots) to make the paradigm inspectable and controllable. This engineering layer does not contradict NL — it makes NL's distributed memory observable and auditable for a product system.

**R6. Reflection and Consolidation Are Core**

The slow reflection path converts lived interaction into durable cognitive change:

- Reads interaction traces, decisions, outcomes, and prediction errors
- Extracts durable lessons, not just summaries
- Produces two types of output: **memory consolidation** (beliefs, open loops, preference traces) and **policy consolidation** (controller priors, strategy preferences, regime weights)
- Runs at `background-slow` timescale, does not block live interaction

In NL terms, reflection is the `background-slow` CMS layer compressing long-window prediction errors into persistent structure. In ETA terms, it is the SSL phase of the slow-scale wake-sleep cycle.

### Credit, modification, and state

**R9. Hierarchical Credit Assignment**

Credit assignment operates at multiple levels: token/utterance, turn, session, long-horizon, and abstract-action. All credit derives from prediction error aggregated at different timescales. Sparse rewards are expected, not edge cases.

ETA's temporal abstraction simplifies credit assignment: each abstract action corresponds to a complete subgoal execution, so prediction error can be attributed at the abstract-action level rather than per-token.

**R10. Self-Modification Must Be Gated and Layered**

Allowed self-modification targets: retrieval weighting, strategy priors, abstract controller parameters, reflection heuristics, memory promotion thresholds. Direct unrestricted mutation of the foundation model during live operation is out of scope.

Gating rules define what can be modified online, what requires background validation, what requires offline retraining, and what requires human review.

**R11. Runtime State Must Expose a Learnable Internal Representation**

The system maintains explicit internal state rich enough for both behavioral control and later reflection: active motives and tensions, candidate strategies, uncertainty, user-state estimates, relationship-state estimates, current regime, and expected outcomes. If the system cannot name and publish its internal state, it cannot learn reliably from it.

---

## Part 3. NL Bridge — Design Implications

This section distills the NL paper into design-relevant claims. It is not a paper restatement.

### Core claims we adopt

1. **Associative memory is the universal primitive.** All components — MLP layers, attention, optimizers, momentum — can be understood as associative memories mapping keys to values, each optimizing an internal objective. This provides a unified lens for designing the system.

2. **Local Surprise Signal (LSS) is the fundamental value.** Training a layer with backpropagation is equivalent to building an associative memory that maps each input to its prediction error. The gradient is not just an optimization tool; it is the *content* being memorized.

3. **Optimizers are associative memories on gradients.** SGD+momentum is a 2-level associative memory. Adam adds adaptive second moments. M3 adds multi-scale momentum. Each optimizer has different memory management characteristics. The choice of optimizer must reflect the patterns of generated gradients.

4. **Nested levels and frequency ordering.** Components are sorted into levels by update frequency and dependencies. Higher level = lower frequency. Knowledge transfers between levels via: direct parametric conditioning, shared backpropagation at different rates, meta-learned initialization, or generation of one level's parameters by another.

5. **CMS is the memory architecture pattern.** A chain of MLP blocks at different update frequencies, each compressing its context into parameters. Anti-forgetting via knowledge backflow from slow to fast layers. Three variants: nested (slow meta-learns fast init), sequential (chain), independent (parallel + aggregate).

### What NL does NOT dictate for our system

- It does not require removing explicit modules — our snapshot-contract architecture is a *runtime layer* implementing NL's paradigm-level distributed memory.
- It does not mandate a specific optimizer — M3 is one instantiation, not the only one.
- Hope is a reference architecture, not mandatory — our system may implement CMS + temporal control differently.

---

## Part 4. ETA Bridge — Architectural Constraints

This section distills the ETA paper into binding architectural constraints.

### Terminology mapping

| Paper term | Repo term | Meaning |
|---|---|---|
| `e_{t,l}` | residual activation / SubstrateSnapshot | Hidden state at layer `l`, step `t` |
| `β_t` | switch gate / switching gate | Continuous gate in [0,1] controlling abstract action persistence |
| `z_t` | controller code / abstract action code | Low-dimensional latent code specifying current abstract action |
| `z̃_t` | candidate code | Newly sampled code before switch gate interpolation |
| `U_t` | residual controller / decoder output | Linear residual stream controller decoded from `z_t` |
| metacontroller | metacontroller | The encoder + switch unit + decoder system |
| subgoal | abstract action (preferred product term) | A temporally-extended meaningful behavior segment |
| internal RL | Internal RL | RL in `z_t` space, with autoregressive model as part of environment |

### Binding constraints from ETA

1. **Two-stage training is non-optional.** SSL discovers structure (Eq.3: action prediction + `α · D_KL` to standard normal). Internal RL exploits it (causal policy `π(z_t | e_{1:t})`). Mixing the stages or skipping SSL leads to degenerate solutions.

2. **Freezing the substrate is necessary.** Rate-distortion analysis shows joint training of base model + metacontroller destroys temporal structure. The substrate must be frozen (or extremely slowly updated) before temporal abstractions can be discovered.

3. **Switching must emerge, not be hardcoded.** The variational bottleneck `α` drives sparse, subgoal-aligned switching patterns. Manual switch thresholds are fallbacks, not the design target. The system should track switch sparsity as a health metric.

4. **Control must be in latent space, not token space.** Internal RL operates on `z_t`, not on generated tokens. Standard token-level RL (e.g., GRPO applied to raw output) fails on hierarchical sparse-reward tasks.

5. **Non-causal → causal transition.** During SSL, the encoder sees the full sequence (non-causal embedding `s(e_{1:T})`). During deployment/RL, only causal information is available. This asymmetry is the core of the variational training — the causal policy must learn to approximate the non-causal posterior without access to future tokens.

---

## Part 5. Integrated Mapping: NL × ETA × EmoGPT

### How the pieces fit together

```mermaid
flowchart TD
    UserInput[User Input] --> SubstrateCapture["Substrate Capture: e_tl"]

    subgraph cogLoop [Cognitive Loop]
        SubstrateCapture --> Temporal["Metacontroller: beta_t, z_t"]
        Temporal --> Decoder["Decoder: z_t -> U_t"]
        SubstrateCapture --> Memory["CMS / MemoryStore"]
        SubstrateCapture --> DualTrack["Dual Track: world + self"]
        DualTrack --> Regime["Regime Selection"]
        Memory --> Regime
    end

    Decoder --> Expression["LLM Expression Layer"]
    Regime --> Expression
    Expression --> Response[Response to User]

    subgraph learnLoop [Learning Loop]
        SubstrateCapture --> Prediction["Predicted Outcome"]
        NextTurn[Next User Turn] --> Actual["Actual Outcome"]
        Prediction --> PredError["Prediction Error / LSS"]
        Actual --> PredError
        PredError --> CreditLedger["Attribution Ledger"]
        PredError --> MemUpdate["Memory Consolidation"]
        PredError --> PolicyUpdate["Policy / Controller Update"]
        PredError --> Reflection["Slow Reflection"]
    end
```

### Key integration principles

1. **Prediction error flows downward, not upward.** Prediction error is computed from the mismatch between expected and actual outcomes. From that primitive, credit records are aggregated, memory is consolidated, controllers are updated, and reflection is triggered. Evaluation scores are readouts, not sources.

2. **Memory, policy, and reflection are all expressions of nested memory update.** In NL terms, they are different associative memories at different frequencies, all driven by the same prediction error signal. In engineering terms, they are distinct modules with distinct owners, publishing their state through immutable snapshots.

3. **The LLM generates text; the cognitive loop generates behavior.** The LLM's role is to turn internal control state into natural language. It does not decide strategy, select regime, or assign credit. Those happen in the layers above it, in the latent controller space.

4. **IQ and EQ are not separate systems.** IQ is reducing prediction error in the world/task track. EQ is reducing prediction error in the self/relationship track. They share the substrate and the temporal abstraction layer but maintain distinct error streams, credit ledgers, and memory paths.

5. **CMS implements NL's frequency ordering; metacontroller implements ETA's temporal abstraction.** These are complementary, not redundant. CMS provides multi-timescale knowledge persistence. The metacontroller provides temporal segmentation and latent control. CMS can enhance the metacontroller's memory (R3); the metacontroller's prediction errors can drive CMS updates (R1).

---

## Part 6. Acceptance Questions

The design is on-track only if the answer to most of these becomes "yes":

1. Does the system expose prediction error explicitly as a first-class runtime object?
2. Does memory update from prediction error at multiple timescales (`online-fast` through `background-slow`)?
3. Does `β_t` produce sparse, meaningful temporal segmentation without manual label supervision?
4. Does Internal RL happen in latent controller space (`z_t`) rather than raw token space?
5. Do world/self tracks produce distinct prediction-error trajectories?
6. Can the system adapt across sessions without full retraining?
7. Can it improve from sparse, delayed social or task outcomes?
8. Can it separate relationship learning from pure task optimization?
9. Can it consolidate experience into durable memory and controller priors?
10. Can it expose enough internal state to support reflection, evaluation, and rollback?
11. Can new adaptive layers be added without destroying module ownership and public contracts?

---

## Non-Goals

This system does not require:

- unrestricted online training of all model parameters
- removing explicit modules in favor of an opaque monolith
- replacing all symbolic or structured state with latent-only representations
- assuming human-level AGI emerges from scaling alone
- treating relationship behavior as merely prompt style
- collapsing ETA's temporal abstraction into token-level RL
- implementing every NL paper construction literally (Hope, M3, etc. are design patterns, not mandatory)

---

## Summary

**NL** provides the system-level doctrine: prediction error as primitive signal, associative memory as universal lens, multi-timescale nested learning, continuum memory.

**ETA** provides the action mechanism: discover temporally-abstract internal controllers through variational SSL, reinforce them in latent space through Internal RL, compose them for hierarchical behavior.

**EmoGPT** applies both to a relationship-aware digital being: dual-track prediction (task + relationship), regime-driven behavioral control, snapshot-first engineering discipline, and bounded self-modification with rollback.

The result is not a better prompt stack, but a bounded learning organism:

- stable at the substrate
- adaptive at the controller layers
- reflective in the background
- memory-rich across timescales
- relationship-aware as a first-class objective
- explainable through explicit contracts
- learning from its own prediction errors
