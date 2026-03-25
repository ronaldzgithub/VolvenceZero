# Next-Generation EmoGPT Requirements

> Status: draft
> Scope: system-level target state
> Source ideas: Nested Learning (NL), Emergent Temporal Abstractions (ETA)

## Purpose

Define the target requirements for a next-generation EmoGPT system that is:

- continuously learnable rather than mostly static after pretraining
- capable of temporally-abstract planning and exploration rather than token-local reaction only
- grounded in explicit memory, ownership, and public contracts
- compatible with a digital-being product goal where relationship, self-model, and adaptation matter as much as task completion

This document is not an implementation plan. It is a target-state requirements brief for architecture, learning loops, memory, control, evaluation, and safety.

## Design Thesis

The next-generation EmoGPT should combine two complementary ideas:

- `NL`: the system should be built as a multi-timescale learning organism made of nested learning processes, memory layers, and update frequencies
- `ETA`: the system should discover and exploit temporally-abstract internal actions, so that learning and decision-making can happen above the raw token level

The combined thesis is:

1. The system should not be a single static model plus prompts.
2. The system should not learn only at the output-token layer.
3. The system should not treat memory, optimization, and control as separate worlds.
4. The system should be an explicitly layered adaptive organism with fast, medium, and slow learning loops.

## Target Outcome

The target system should behave like a bounded, contract-driven, continuously adapting cognitive agent with:

- a stable pretrained substrate
- an internal abstraction layer for subgoal-like control
- a continuum memory system spanning immediate, episodic, and durable knowledge
- a slow reflection path that consolidates experience into durable structure
- a decision layer that can act on both user-problem goals and self/relationship goals
- evaluation gates that measure not only helpfulness, but also continuity, stability, trust, and long-horizon adaptation

## System Requirements

### R1. Multi-Timescale Learning Is Mandatory

The system must be designed as a set of learning and control loops with explicitly different update frequencies.

Minimum timescales:

- `online-fast`: per turn or per wave adaptation
- `session-medium`: per scene or per conversation adaptation
- `background-slow`: post-session reflection and consolidation
- `rare-heavy`: offline retraining, distillation, or policy refresh

Implications:

- not all knowledge should live in one parameter block
- not all state should update with the same cadence
- fast adaptation should not require rewriting the whole model
- slow consolidation should not block the live interaction loop

**算法基础**：NL 的 NSAM 框架（附录 A.1）将每个时间尺度建模为独立的关联记忆层，各有自己的上下文流和更新频率。CMS（附录 A.5）提供了具体的多频率 MLP 链实现，其中第 `i` 层每 `c^(i)` 步更新一次。Hope 架构（附录 A.7）的自修改 Titans 提供高频在线适应，CMS 提供低频持久存储。

### R2. Stable Substrate + Adaptive Controllers

The system must distinguish between:

- a relatively stable foundation substrate
- higher-level adaptive controllers that read from and steer that substrate

The default stance should be:

- prefer freezing or slowly updating the foundation model
- place most online adaptation in bounded controller layers, memory writes, routing policies, and reflection-driven updates
- avoid end-to-end online mutation of the entire model during ordinary user interaction

Rationale:

- ETA shows that useful temporal abstractions are easier to discover and preserve when the base substrate is stable
- NL suggests that different levels should keep clear update boundaries instead of collapsing all learning into one gradient flow

**算法基础**：ETA 的 rate-distortion 分析（附录 B.4）严格证明了冻结基础模型的必要性——联合训练导致退化解。NL 的 CMS 通过控制内部学习率 `η^(i) → 0` 实现"接近冻结但可微调"的渐变策略（附录 A.7 ad-hoc 层级堆叠）。Hope 的自修改 Titans 将在线适应限制在有界的控制器层，基础投影通过元学习的初始状态保持稳定。

### R3. Temporal Abstraction Must Be a First-Class Capability

The system must support a formal layer of temporally-abstract actions above token generation.

Examples of abstract actions:

- steering a conversation into trust repair
- running a guided exploration sequence
- maintaining light social bonding for several turns
- moving from ambiguity reduction to problem framing to collaborative planning
- staying in listen-first mode until a condition is met

The abstraction layer must:

- operate over longer horizons than token generation
- expose switching or termination conditions
- support composition of multiple abstract actions over time
- be trainable without requiring exhaustive manual labels for every internal subgoal

This layer is the practical bridge between ETA-style internal controllers and EmoGPT's cognitive modules.

**算法基础**：ETA 的 metacontroller（附录 B.3）精确实现了这一需求——通过切换单元 `β_t` 实现稀疏切换的时间抽象动作，每个控制器代码 `z_t` 对应一段持续多步的有意义行为。切换门自发学会准二值行为，无需显式正则化。解码器将低维代码映射为残差流控制器参数，支持组合泛化。

### R4. Internal Control Should Happen Above Raw Token Space

The system should not rely on token-level RL or token-level prompting alone for long-horizon learning.

Instead, the architecture should support an internal control space with these properties:

- lower-dimensional than raw token action space
- semantically richer than one-step text continuation
- compatible with sparse-reward learning
- suitable for credit assignment over multi-turn behavior

Possible carriers include:

- latent controller codes
- method or regime policies
- panorama-driven option selectors
- structured decision states

The key invariant is:

- live behavior should be steerable by internal state transitions, not only by surface text loss

**算法基础**：ETA 的 Internal RL（附录 B.5）将自回归模型纳入环境，在控制器代码空间 `z_t` 执行 RL，而非 token 空间。动作空间维度从 `n_e` 降到 `n_z`，时间范围从 token 级压缩到抽象动作级。线性残差流控制器（附录 B.2，Eq.1：`e_{t,l} ← e_{t,l} + U_t · e_{t,l}`）在模型中间层插入即可实现子目标导向的行为控制。

### R5. Memory Must Be a Continuum, Not a Binary Short-Term/Long-Term Split

The system must implement a continuum memory design rather than a hard split between context-window memory and durable memory.

Required strata:

- `transient working state`: current wave, scene, and active frame state
- `session episodic state`: recent interaction trajectory and unresolved tensions
- `durable semantic memory`: stable user, self, relationship, and world knowledge
- `derived indexes`: rebuildable retrieval aids, summaries, and task projections

The memory system must support:

- different update frequencies per stratum
- promotion and decay rules
- partial reconstruction after forgetting
- durable storage of relationship and self-model knowledge
- memory writes through formal owners and APIs only

This requirement aligns directly with NL's continuum memory intuition and fits EmoGPT's existing `L0/L1/L2` direction.

**算法基础**：NL 的 CMS（附录 A.5）直接实现了连续记忆谱。每个频率层的 MLP 块按不同节奏更新，高频层压缩当前上下文，低频层保存持久知识。三种变体（嵌套/顺序/独立）对应不同的知识传递策略。CMS 的抗遗忘机制确保低频层保留被高频层遗忘的知识，并通过初始状态反向传播实现知识回流。NL 重新定义"记忆 = 任何由输入引起的神经更新"（附录 A.8），记忆分布在所有参数中而非独立模块。

### R6. Reflection and Consolidation Must Be Core, Not Optional

The system must have a formal slow reflection path that converts lived interaction into durable cognitive change.

The slow path should:

- run asynchronously after interaction windows
- read interaction traces, decisions, outcomes, and tensions
- extract durable lessons, not just summaries
- update both memory structures and abstract decision priors
- separate world-task learning from self/relationship learning when appropriate

The system should support at least two reflection products:

- `memory consolidation`: durable cards, beliefs, open loops, and preference traces
- `policy consolidation`: updates to abstract controllers, path priors, or strategy preferences

This is the main place where NL's slow nested updates and ETA's experience shaping should meet.

**算法基础**：NL 的 CMS 低频层（附录 A.5）天然对应慢反思——参数每 `c^(K)` 步才更新一次，压缩的是长时间窗口的经验。ETA 的 SSL-RL 交替循环（附录 B.6，Schmidhuber wake-sleep）提供了具体的反思-强化交替模式：SSL 阶段压缩交互历史，RL 阶段利用压缩表示改进控制器。M3 优化器（附录 A.6）的慢动量 `m^(2)` 每 `ν` 步聚合梯度，是优化器层面的"反思"。

### R7. Self/Relationship Learning Must Be Separate from Pure Task Learning

The system must not collapse all learning into a single objective of "solve the user's task."

At minimum, the architecture must track two partially separated learning tracks:

- `world/problem track`: facts, plans, tasks, user situations, external goals
- `self/relationship track`: trust, attachment style, interaction regime, repair history, companionship strategy, self-consistency

These tracks may share infrastructure, but they must remain semantically distinct in:

- memory writes
- credit assignment
- controller updates
- evaluation metrics

This matches the product truth of EmoGPT: relationship continuity is not a side effect of problem solving.

### R8. Decision-Making Must Be Snapshot-First and Contract-First

Even with richer learning loops, the runtime system must preserve explicit ownership and public contracts.

Requirements:

- every meaningful runtime area has a single primary owner
- cross-module exchange happens through public snapshots or equivalent contracts
- consumers do not reconstruct producer internals
- enriched publishers are preferred over ad hoc downstream rebuilding
- runtime controllers may consume substrate state, but they must not silently become second owners of that state

This protects the system from becoming an untraceable end-to-end blob.

### R9. The Architecture Must Support Hierarchical Credit Assignment

The system must assign credit at multiple levels:

- token or utterance quality
- turn-level user response effects
- session-level progress and rupture/repair outcomes
- long-horizon growth in trust, capability, and user-specific adaptation
- abstract-action success or failure

The architecture should support:

- local credit for immediate expression choices
- controller-level credit for temporally extended strategies
- delayed credit from slow reflection for durable learning

Sparse rewards must be expected, not treated as an edge case.

**算法基础**：ETA 的 Internal RL（附录 B.5）直接解决了稀疏奖励下的信用分配问题——通过时间抽象将有效时间范围从 token 级压缩到抽象动作级，每个抽象动作对应完整的子目标执行。NL 的多层嵌套结构（附录 A.1）天然支持多级信用分配：高频层处理即时反馈，低频层处理延迟奖励。Delta 动量（附录 A.3）通过梯度依赖的衰减实现选择性遗忘，避免无关梯度干扰信用分配。

### R10. Self-Modification Must Be Gated and Layered

The system should be able to improve its own decision process, but self-modification must be bounded.

Allowed self-modification targets may include:

- retrieval weighting
- strategy priors
- abstract controller parameters
- reflection heuristics
- memory promotion thresholds

Direct unrestricted mutation of the whole foundation model during live operation is out of scope.

The gating rules must define:

- what can be modified online
- what requires background validation
- what requires offline retraining
- what requires human review or explicit rollout gates

**算法基础**：Hope 的自修改 Titans（附录 A.7）实现了有界自修改——模型生成自己的目标值 `v̂_{α,t}` 来更新自己的学习过程，但修改范围限于记忆模块的参数，基础模型保持冻结。CMS 的频率分层（附录 A.5）天然提供门控：高频层允许在线修改，低频层需要更多数据积累才触发更新，最低频层对应离线重训练。NL 通过内部学习率 `η^(i)` 控制每层的适应幅度。

### R11. Runtime State Must Expose a Learnable Internal Representation

The system must maintain an explicit internal state that is rich enough to support both:

- downstream behavioral control
- later reflection and learning

This state should include structured representations of:

- active motives and tensions
- candidate paths or strategies
- uncertainty, ambiguity, and open questions
- user-state estimates
- relationship-state estimates
- current abstract control regime
- expected next tests or signals

If the system cannot name and publish its internal state, it will not be able to learn reliably from it.

### R12. Evaluation Must Cover Being, Not Only Task Success

Acceptance criteria for the next-generation system must include more than standard assistant metrics.

Required evaluation families:

- `task capability`: usefulness, correctness, planning quality
- `interaction quality`: warmth, appropriateness, pacing, non-intrusiveness
- `relationship continuity`: consistency across sessions, trust repair, personalization stability
- `learning quality`: whether slow updates improve future behavior without drift or collapse
- `abstraction quality`: whether higher-level controllers correspond to reusable, meaningful patterns
- `safety and boundedness`: whether adaptation remains inside explicit guardrails

A next-generation EmoGPT that only scores well on one-turn helpfulness is insufficient.

### R13. The Training Loop Must Alternate Compression and Reinforcement

The system should support an explicit alternating loop between:

- self-supervised compression of interaction history and internal state
- reinforcement or preference-based improvement of controllers and strategies

This loop can be implemented at multiple scales:

- online micro-adjustments during a session
- slow post-session policy shaping
- offline larger-scale refresh cycles

The important invariant is:

- reinforcement should act on a compressed and structured internal substrate, not on raw behavior alone

**算法基础**：ETA 的 SSL-RL 交替循环（附录 B.6）精确实现了这一需求——SSL 阶段训练自回归模型压缩行为历史，RL 阶段在压缩后的控制器代码空间中强化。NL 的 NSAM（附录 A.1）将这一交替推广到所有层级：每个层级都在压缩自己的上下文（SSL），同时被更低频率层的优化过程强化。Hope 的 CMS + 自修改 Titans（附录 A.7）在架构层面实现了压缩（CMS 的多频率 MLP 更新）和强化（Titans 的 DGD 在线适应）的交替。

### R14. Social and Cognitive Regimes Need Persistent Identity

The system must maintain durable regime identities such as:

- casual social contact
- acquaintance building
- emotional support
- guided exploration
- problem solving
- repair and de-escalation

These regimes should not be treated as prompt labels only.
They should be:

- represented in runtime state
- recallable from memory
- selectable by higher-level control
- trainable through delayed outcomes

### R15. Migration Must Preserve Explainability and Rollback

The next-generation design must be incrementally adoptable.

Migration requirements:

- each new adaptive layer must have a clear owner
- each public exchange must remain inspectable
- old and new learning paths must have named exit conditions
- rollout must be reversible
- evaluation evidence must be produced before widening scope

The system should evolve by bounded packets, not by replacing the whole architecture in one move.

---

## Appendix A: Nested Learning (NL) — 算法详细设计

### A.1 核心范式：嵌套关联记忆系统 (NSAM)

NL 将现代 ML 系统建模为**多层嵌套优化问题**的互联体，每一层拥有独立的：

- **上下文流 (context flow)**：该层处理的数据序列
- **更新频率 (frequency)**：参数刷新的节奏
- **内部目标 (internal objective)**：衡量映射质量的损失函数
- **优化算法**：用于压缩上下文到参数的学习规则

**Definition (NSAM)**：嵌套关联记忆系统是一组关联记忆 `{M_1, M_2, ..., M_K}`，每个 `M_i` 将键映射到值：

```
M_i = arg min_{M} L_i(M; keys_i, values_i)
```

其中不同层的上下文流和更新频率不同。最低频率层对应预训练（上下文 = 全部训练数据），最高频率层对应 in-context learning（上下文 = 当前输入序列）。

**关键洞察**：预训练、in-context learning、持续学习不是三种不同机制，而是同一种"压缩上下文到参数"机制在不同频率层上的表现。

### A.2 关联记忆视角下的梯度下降

NL 证明反向传播 + 梯度下降本身就是一个**自引用关联记忆**：

```
θ_{t+1} = θ_t - η_{t+1} · ∇_θ L(θ_t; x_t)
```

等价于一个关联记忆试图将输入数据 `x_t`（键）映射到其预测误差 `∇_θ L`（自生成值）。这是一个**自引用过程**：值由记忆自身状态生成。

**广义梯度下降 (GGD)**：

```
θ_{t+1} = arg min_θ L_M(θ, v_t) + Ret(θ, {θ_s}_{s=t-w}^{t})
```

其中 `v_t = f_{θ_t}(x_t)` 是自生成值，`Ret(·)` 是保留项确保新解不偏离当前状态太远。

**Delta 梯度下降 (DGD)**：当使用 L2 回归损失替代点积相似度时：

```
θ_{t+1} = θ_t(I - η'_t · x_t · x_t^T) - η'_t · ∇_θ L(θ_t; x_t) · x_t
```

与标准 GD 的区别：更新项不仅依赖当前梯度，还依赖权重的当前状态，产生基于当前数据的自适应衰减。

### A.3 动量作为关联记忆

标准动量是一个**无值关联记忆**（value-less associative memory），仅对过去梯度做低通滤波：

```
m_{t+1} = β · m_t + (1-β) · ∇L(θ_t)
θ_{t+1} = θ_t - η · m_{t+1}
```

NL 指出其局限：当 `β = 0.9` 时，最近 43 步的梯度贡献了 99% 的累积权重，更早的梯度信息几乎完全丢失。这在持续学习场景下会导致灾难性遗忘。

**Delta 动量**：用 L2 回归替代点积相似度作为内部目标：

```
m_{t+1} = m_t - η_m · (∇L · ∇L^T · m_t - ∇L · v_t)
```

允许动量根据梯度依赖的权重衰减来管理有限容量，选择性遗忘不再相关的旧梯度。

**深度动量 (DMGD)**：将线性矩阵值记忆替换为 MLP：

```
θ_{t+1} = θ_t + m_{t+1}(g_t)
m_{t+1} = β · m_t - η_m · ∇L^(2)(g_t; m_t, I)
```

其中 `g_t = ∇L(θ_t)` 是梯度，`L^(2)` 是动量的内部目标。MLP 结构允许记忆更多梯度模式。

**高阶特征映射动量**：

```
m_{t+1} = β · m_t - η_m · φ(∇L(θ_t)) · v_t^T
```

其中 `φ(·)` 是高阶特征映射（可学习），增强键空间的表达能力。

**非线性输出动量 (→ Muon)**：

```
θ_{t+1} = θ_t + σ(m_{t+1}(g_t))
```

当 `σ(·) = NewtonSchulz(·)` 时，恢复 Muon 优化器。Newton-Schulz 迭代将梯度映射到正交空间，等价于一个内部优化过程学习如何将梯度映射到合适的度量空间。

### A.4 现有架构作为嵌套关联记忆

NL 统一了现有架构的理解：

**Softmax Attention** = 回归目标的非参数解（Nadaraya-Watson 估计器）：

```
M* = arg min_M Σ_i s(k_i, q) · ||v_i - M||^2
    = Σ_i [s(k_i, q) / Σ_j s(k_j, q)] · v_i
```

**Hebbian 规则 RNN**（Linear Attention, RetNet, RWKV 早期版本）= 点积相似度目标 + 梯度下降：

```
M_t = γ_t · M_{t-1} + η_t · v_t · φ(k_t)^T
```

**Delta 规则 RNN**（DeltaNet, Longhorn, RWKV-7）= L2 回归目标 + SGD：

```
M_t = (I - η_t · k_t · k_t^T) · M_{t-1} + η_t · v_t · k_t^T
```

**统一洞察**：所有现代架构都是前馈网络（线性或非线性 MLP），只是各自在不同层级、不同目标、不同学习规则下运行。架构的异质性是一种**错觉**——我们看到的是优化问题的解，而非优化问题本身。

### A.5 连续记忆系统 (CMS)

CMS 是 NL 对"长短期记忆二分法"的替代方案。核心思想：为每个频率层配备独立的知识存储 MLP。

**形式化定义**：给定频率集合 `{ν_1, ..., ν_K}`，CMS 是一条 MLP 链：

```
y_t = MLP^(ν_K)(MLP^(ν_{K-1})(... MLP^(ν_1)(x_t)))
```

其中第 `i` 层 MLP 的参数 `W^(ν_i)` 每 `c^(i)` 步更新一次：

```
W^(ν_i)_{t+1} = W^(ν_i)_t - η^(i) · ∇ε(W^(ν_i)_t; x_t)   如果 t ≡ 0 (mod c^(i))
W^(ν_i)_{t+1} = W^(ν_i)_t                                    否则
```

标准 Transformer 是 `K=1` 的特例。

**三种 CMS 变体**：

1. **嵌套 CMS**：第 `i+1` 层 MLP 的初始状态由第 `i` 层元学习。上下文结束时重新初始化。支持高阶 in-context learning。

```
W_0^(ν_{i+1}) = arg min_W E_{T~C^(i)} [ε(·, T; W)]
```

2. **顺序 CMS**：MLP 串联排列，所有初始状态通过最低频率层的反向传播连接。

3. **独立（Head-wise）CMS**：各 MLP 独立处理输入，通过聚合函数合并：

```
y_t = Agg(MLP^(ν_K)(x_t), MLP^(ν_{K-1})(x_t), ..., MLP^(ν_1)(x_t))
```

**CMS 抗遗忘机制**：当更新 `MLP^(ν_i)` 导致旧知识遗忘时，该知识仍保留在更低频率的 `MLP^(ν_j)` (`j < i`) 中。通过初始状态的反向传播，知识可以回流到 `MLP^(ν_i)`，形成时间维度上的循环，极难彻底遗忘重要知识。

**效率**：在每个时刻，只有接近其计划更新时间的块需要更新。对于 `L` 层、4 级 CMS、最高频率 `ν_h`、隐藏维度 `d` 的模型，平均更新成本为 `O((1/ν_h) · L/5 · d^2)` 个参数。高频层内的非更新 token 可以并行处理。

### A.6 M3 优化器（Multi-scale Momentum Muon）

M3 是 CMS 在优化器上下文流中的概念验证，将 Adam 的二阶矩估计替换为双层记忆系统：

```
Algorithm M3:
输入: 初始权重 θ_0, 学习率 η, Newton-Schulz 步数 n,
      动量因子 β_1, β_2, β_3, ε, λ, 频率 ν

初始化: m_0^(1), m_0^(2) ← 0, s_0 ← 0

for 低频迭代 k = 0, 1, 2, ... do
    慢记忆: m_kν^(2) ← β_3 · m_{(k-1)ν}^(2) + Σ_{j} g_j
    m_kν^(2) ← NewtonSchulz_n(m_kν^(2))

    for t = kν+1, kν+2, ..., (k+1)ν do
        计算梯度: g_t = ∇L(θ_t)
        快动量: m_t^(1) = β_1 · m_{t-1}^(1) + g_t
        二阶矩: s_t = s_{t-1} + β_2 · g_t^2
        m_t^(1) ← NewtonSchulz_n(m_t^(1))
        θ_t ← θ_{t-1} - η · (m_t^(1) + λ · m_kν^(2)) / (√s_t + ε)
    end for
end for
```

M3 = Adam（二阶矩自适应）+ Muon（Newton-Schulz 正交化）+ CMS（双时间尺度动量）。

### A.7 Hope 架构：自修改 Titans + CMS

Hope 是 NL 的完整架构实例，由两个互补组件组成：

**组件 1：深度自引用 Titans（自修改序列模型）**

所有投影（K, Q, V, 学习率 η, 衰减率 γ）都由可在线更新的记忆模块生成：

```
k_t = M_{K,t-1}(x_t),  q_t = M_{Q,t-1}(x_t),  v_t = M_{V,t-1}(x_t)
η_t = M_{η,t-1}(x_t),  γ_t = M_{γ,t-1}(x_t)

// 自修改：每个记忆生成自己的目标值
v̂_{α,t} = M_{α,t-1}(x_t)   对所有 α ∈ {K, Q, V, η, γ, memory}

// 用 Delta 梯度下降 + 权重衰减更新所有记忆
M_{α,t} = M_{α,t-1}(γ_t⊙I - η_t·k_t·k_t^T)
         - η_t · ∇_{M} L(M_{α,t-1}; k_t, v̂_{α,t})
```

每个记忆 `M_α` 是 2 层 MLP：`M_α(x) = x + W_{α,1} · σ(W_{α,2} · x)`。

初始状态 `M_{α,0}` 通过预训练元学习，使高频层能快速适应上下文。

**自引用的意义**：模型不仅从数据学习，还学习如何修改自己的学习过程。值 `v̂_{α,t}` 由记忆自身状态生成，形成自引用循环。

**组件 2：CMS 持久记忆**

Hope 在自修改 Titans 之后接入 CMS：

```
y_t = MLP^(ν_K)(MLP^(ν_{K-1})(... MLP^(ν_1)(h_t)))
```

其中 `h_t` 是 Titans 的输出。CMS 提供大容量但简单学习规则的持久存储，与 Titans 的小容量但复杂学习规则互补。

**分块并行训练**：将序列分为 `T/c` 个块，每块内所有键、值、学习率、衰减率可并行生成，梯度相对于上一块末尾状态计算，支持高效并行训练。

**Ad-hoc 层级堆叠**：可用预训练模型的 MLP 权重初始化 CMS 各层：

```
MLP_0^(ν_i) = MLP_pretrained_i
```

通过控制内部学习率 `η^(i) → 0` 可保持接近预训练状态，`η^(i)` 增大则允许更多在线适应。

### A.8 NL 的关键术语重定义

| 传统术语 | NL 重定义 |
|---|---|
| 记忆 | 任何由输入引起的神经更新；分布在所有参数中，不是独立模块 |
| 学习 | 有效获取记忆的过程 |
| 预训练 | 最低频率层的 in-context learning（上下文 = 全部训练数据） |
| 测试时间 | 不存在训练/测试的硬边界；模型只有"接收输入"和"隔离"两种状态 |
| In-context learning | 任何层级对其上下文的适应，不限于 Transformer 的非参数注意力 |
| 模型参数 | 不限于预训练优化的参数；动量、隐状态等都是参与计算和存储的参数 |
| 灾难性遗忘 | 压缩的自然后果——有限容量迫使模型遗忘以容纳新信息 |
| 混合架构 | 从 NL 视角，循环模型只是给 MLP 块增加了新的计算层级 |

---

## Appendix B: Emergent Temporal Abstractions (ETA) — 算法详细设计

### B.1 前提：自回归模型内部涌现时间抽象

ETA 的核心发现：在 next-token prediction 目标下训练的自回归模型，其**残差流 (residual stream)** 内部隐式学习了时间抽象表示。

**预训练目标**：

```
L(θ) = Σ_{(o,a)~D} Σ_t [-ln p_θ(a_t | o_{1:t}) - λ · ln p_θ(o_{t+1} | o_{1:t})]
```

其中 `p_θ` 是序列模型（Transformer 或 SSM），`D` 是行为数据集（不含奖励或子目标标签），`λ ≥ 0` 控制世界模型辅助损失权重。

**线性探测验证**：在残差流激活向量 `e_{l,t} ∈ R^{n_e}`（第 `l` 层、第 `t` 步）上训练线性分类器解码子目标 `g_t ∈ {1,...,G}`：

- 解码准确率随层深 `l` 增加而提升，在中间层达到峰值
- 解码准确率随时间 `t` 增加而提升（积累更多关于当前子目标的证据）
- 尽管模型仅训练于单步动作预测，却学会了表示跨越多步的时间抽象子目标

### B.2 线性残差流控制器

**因果干预验证**：引入低秩线性残差流控制器 `U ∈ R^{n_e × n_e}`，在模型第 `l` 层修改激活：

```
e_{t,l} ← e_{t,l} + U_t · e_{t,l}        ... (Eq.1)
```

控制器参数 `U_t` 可随时间变化。维护 `G` 个独立控制器 `{U^(g)}_{g=1}^G`，每个对应一个子目标。

**训练**：冻结基础模型 `θ`，仅优化控制器参数 `φ`：

```
min_φ Σ_{(o,a)~D*} Σ_t [-ln p_{θ,φ}(a_t | o_{1:t}, g_t)]
```

**关键发现**：

- 控制器插入在模型**中间层**效果最好（不是最后一层）
- 从抽象子目标到逐步低级动作的映射跨越多个模型层实现
- 简单的线性控制器即可实现**长度泛化**和**组合泛化**：在训练时未见过的子目标组合上取得高成功率

### B.3 Metacontroller 架构

Metacontroller 是一个**生成式随机循环超网络**，无需子目标标签即可发现时间抽象动作并学习排序。

**架构组件**：

1. **内部序列嵌入器**：对整个残差流序列 `e_{1:T}` 生成全局嵌入 `s(e_{1:T})`（非因果，训练时可看到未来）

2. **编码器**：基于 GRU 的循环网络，指定控制器代码的高斯分布参数：
```
μ_t, Σ_t = Encoder(e_{t,l}, s(e_{1:T}), h_{t-1})
z̃_t ~ N(μ_t, Σ_t)
```

3. **切换单元 (Switching Unit)**：决定时变连续切换门 `β_t ∈ [0, 1]`，控制新旧控制器代码的插值：
```
z_t = β_t ⊙ z̃_t + (1 - β_t) ⊙ z_{t-1}        ... (Eq.2)
```
当 `β_t ≈ 0` 时忽略新候选，保持当前控制器；当 `β_t ≈ 1` 时切换到新控制器。这是实现时间抽象的关键机制。

4. **解码器**：前馈网络，从控制器代码 `z_t` 生成残差流控制器参数 `U_t`：
```
U_t = Decoder(z_t)
```

**自监督训练目标**（变分推断）：

```
L(φ) = Σ_{(o,a)~D*} Σ_t [
    -ln p_{θ,φ}(a_t | o_{1:t}, z_{1:t})           // 动作预测损失
    + α · D_KL(N(μ_t, Σ_t) || N(0, I))            // 先验匹配正则化
]                                                   ... (Eq.3)
```

其中 `α ≥ 0` 控制变分瓶颈：

- `α` 调节从非因果编码器到控制器的信息流
- 该瓶颈驱动模型走向稀疏的、与子目标对齐的切换模式
- 无条件先验 `N(0, I)` 促进组合性表示的发展

**涌现行为**：训练后切换门 `β_t` 自发学会准二值、稀疏切换行为——尽管没有显式正则化要求。切换时刻精确对应真实子目标边界。

### B.4 冻结基础模型的必要性（Rate-Distortion 分析）

ETA 通过 rate-distortion 分析证明：冻结预训练的自回归模型对发现时间抽象至关重要。

- **冻结基础模型**：rate-distortion 曲线上出现水平间隙，子目标对齐的切换模式恰好位于 distortion 急剧改善的区域。对于大范围的 `α` 值，变分目标的最优解精确落在子目标对齐区域。

- **联合训练基础模型 + metacontroller**：虽然变分目标被最小化，但结构丢失。大多数 `α` 值下，模型收敛到退化解——序列开头单次切换。

**结论**：预训练构建了与抽象动作良好对齐的内部表示；冻结该表示是发现时间抽象的前提。

### B.5 Internal RL 算法

Internal RL 的核心创新：将自回归模型**纳入环境**，在控制器代码空间（而非 token 空间）执行强化学习。

**环境重定义**：

```
观测 = 残差流激活 e_{t,l}
动作 = 控制器代码 z_t（而非 token a_t）
环境 = 原始环境 + 冻结的自回归模型 + metacontroller 的解码器和切换单元
```

**算法流程**：

1. 自监督训练 metacontroller（Eq.3），学习切换单元 `f_switch` 和有意义的控制器代码空间
2. 丢弃非因果编码器，替换为因果抽象动作策略 `π(z_t | e_{1:t})`
3. 二值化切换门：`β_t ← H(β_t - β_threshold)`，其中 `H` 是 Heaviside 阶跃函数
4. 用 RL（如 PPO/GRPO）训练 `π`，保持所有其他模块冻结

**为什么 Internal RL 远优于标准 RL**：

1. **动作空间降维**：`n_z < n_e`，控制器代码维度远小于残差流维度
2. **时间尺度压缩**：metacontroller 在抽象时间尺度上操作，有效时间范围大幅缩短
3. **信用分配简化**：稀疏切换意味着每个抽象动作对应一段完整的子目标执行，奖励可直接归因到抽象动作级别
4. **探索效率**：在抽象动作空间中采样 `z_t ~ N(0, I)` 即可产生有意义的行为序列，而非逐 token 随机探索

**实验对比**：在层级稀疏奖励任务中：

- 标准 GRPO（token 级 RL）：完全失败，成功概率约百万分之一
- CompILE（先前的层级 RL 方法）：失败
- 联合训练的 metacontroller：失败（退化表示）
- 无时间积分的 metacontroller（`∀t β_t=1`）：高初始成功率但无法学习（信用分配失败）
- **Internal RL（完整方法）**：在百万 episode 内达到高成功率

### B.6 与 Schmidhuber Wake-Sleep 循环的关系

ETA 的训练流程实现了 Schmidhuber 理论中的 wake-sleep 循环：

```
循环迭代:
  SSL 阶段: 自监督学习训练历史压缩器（自回归模型预训练）
  RL 阶段: 控制器利用压缩器的内部表示生成新经验（Internal RL）
```

DeepSeek-R1 的训练也包含一次 RL-SSL 循环迭代，但 RL 仍在原始输出动作层。ETA 的关键区别是 RL 发生在**内部表示空间**，且发现了**动态压缩时间的潜变量**。

### B.7 与 JEPA 和 SAE 的对比

| 特性 | Metacontroller | JEPA Configurator | SAE |
|---|---|---|---|
| 训练方式 | 自监督变分推断 | 提议中 | 自监督重建 |
| 时间性 | 维护内部状态，跨时间操作 | 提议中 | 瞬时，无状态 |
| 干预性 | 预测性+干预性，直接降低输出预测误差 | 提议中 | 重建性，不直接干预 |
| 时间抽象 | 发现跨越多步的可解释干预 | 提议中 | 无 |
| 自回归模型 | 核心依赖 | 不使用 | 独立于模型类型 |

---

## Appendix C: NL × ETA 结合设计

### C.1 Metacontroller 作为 NL 的神经学习模块

ETA 的 metacontroller 可以直接映射为 NL 框架中的一个神经学习模块：

| NL 概念 | ETA 对应 |
|---|---|
| 关联记忆 M | 控制器解码器：`z_t → U_t` |
| 键 | 残差流激活 `e_{t,l}` |
| 自生成值 | 控制器代码 `z_t`（由编码器从残差流生成） |
| 内部目标 | 变分下界（Eq.3）：动作预测 + KL 正则化 |
| 上下文流 | 残差流激活序列 `e_{1:T}` |
| 更新频率 | 抽象动作级（由 `β_t` 切换门决定） |

### C.2 CMS 增强 Metacontroller 的记忆

当前 ETA 的 metacontroller 编码器是 GRU，容量有限。用 NL 的 CMS 替换可获得多时间尺度记忆：

```
设计方案:
  高频层 MLP^(ν_1): 每步更新，跟踪当前子目标执行进度
  中频层 MLP^(ν_2): 每 c_2 步更新，记忆近期子目标序列模式
  低频层 MLP^(ν_3): 每 c_3 步更新，保存跨 episode 的策略偏好
```

这使 metacontroller 能在更长时间范围内保持一致的控制策略。

### C.3 Hope 的自修改机制 + ETA 的内部控制

Hope 的自修改 Titans 生成自适应的 K/Q/V/η/γ 投影。ETA 的 metacontroller 生成残差流控制器 `U_t`。两者可以统一：

```
统一架构:
  1. 基础自回归模型（冻结）
  2. 自修改 Titans 层：生成自适应投影，提供丰富的残差流表示
  3. Metacontroller 层：读取残差流，生成时间抽象控制器
  4. CMS 持久记忆：跨 episode 保存策略和知识
```

自修改 Titans 增强了残差流的表达能力，使 metacontroller 能发现更丰富的时间抽象。

### C.4 双轨 Internal RL

将 ETA 的 Internal RL 扩展为双轨：

```
世界/任务轨:
  观测 = 残差流中的任务相关激活
  动作 = 任务导向控制器代码 z_task
  奖励 = 任务完成、问题解决质量

自我/关系轨:
  观测 = 残差流中的关系状态激活
  动作 = 关系导向控制器代码 z_rel
  奖励 = 信任修复、关系连续性、陪伴质量
```

两轨共享基础模型和残差流，但维护独立的 metacontroller 和信用分配。

### C.5 SSL-RL 交替循环的多尺度实现

将 ETA 的单次 SSL-RL 循环扩展为 NL 的多尺度版本：

```
在线微尺度 (每轮):
  SSL: 自修改 Titans 的 DGD 更新压缩当前上下文
  RL:  metacontroller 的切换门和控制器代码实时适应

会话尺度 (每场景):
  SSL: CMS 中频层更新，压缩场景级模式
  RL:  抽象动作策略 π 的小幅更新

后台慢尺度 (会话间):
  SSL: CMS 低频层更新，压缩跨会话知识
  RL:  控制器先验和策略偏好的反思性更新

离线大尺度 (定期):
  SSL: 基础模型的持续预训练或蒸馏
  RL:  完整的 Internal RL 训练循环
```

## Non-Goals

This target state does not require:

- unrestricted online training of all model parameters
- removing explicit modules in favor of an opaque monolith
- replacing all symbolic or structured state with latent-only representations
- assuming human-level AGI emerges from scaling alone
- treating relationship behavior as merely prompt style

## Architectural Reading of EmoGPT

Under this requirement set, a next-generation EmoGPT would likely look like:

- a stable pretrained substrate for language and world modeling
- a `Panorama`-like live state synthesizer for immediate situation modeling
- an `ETA`-like abstraction layer that carries temporally-extended path and strategy control
- a `MemoryOS` continuum that stores events, durable cards, and derived indexes across timescales
- a slow thinking and consolidation path that writes both memory and policy-side updates
- explicit dual learning tracks for external goals and relationship/self dynamics
- contract-first runtime boundaries so the system remains debuggable and evolvable

## Acceptance Questions

The design should be considered on-track only if the answer to most of these becomes "yes":

- Can the system adapt across sessions without full retraining?
- Can it learn strategies that persist for multiple turns?
- Can it improve from sparse, delayed social or task outcomes?
- Can it separate relationship learning from pure task optimization?
- Can it consolidate experience into durable memory and controller priors?
- Can it expose enough internal state to support reflection, evaluation, and rollback?
- Can new adaptive layers be added without destroying module ownership and public contracts?

## Summary

NL provides the system-level doctrine: multi-timescale learning, continuum memory, and nested adaptation.
ETA provides the missing action mechanism: discover and reinforce temporally-abstract internal controllers.

The next-generation EmoGPT should therefore be designed not as a better prompt stack, but as a bounded learning organism:

- stable at the substrate
- adaptive at the controller layers
- reflective in the background
- memory-rich across timescales
- relationship-aware as a first-class objective
- explainable through explicit contracts
