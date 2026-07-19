# DMP-SNN Neuromorphic Fast-Slow Memory Analysis

> Source: Nature Machine Intelligence, 2026, "Algorithm-hardware co-design of neuromorphic networks with dual memory pathways"  
> URL: https://www.nature.com/articles/s42256-026-01255-3  
> VZ relevance: R1, R3, R4, R5, R6, R8, R15  
> Status: research note, not runtime contract

## 一句话判断

这篇文章的重要性不在于它又提出了一个 SNN block，而在于它把一个 VZ 长期坚持的原则用 neuromorphic 体系做成了硬证据：

> 长程时间能力不必来自 dense recurrence、完整历史缓存或更大的模型；它可以来自一个显式、低维、局部、共享的 slow state，并通过独立 fast path / slow path 的工程镜像获得部署效率。

对 VolvenceZero 来说，它不是近期要引入 SNN 或 neuromorphic hardware 的信号，而是给 R1/R5/R3/R4 提供了一个非常干净的外部证据：**慢状态必须显式化、低秩化、owner-local 化；快路径继续保持事件驱动和稀疏响应；两者之间是调制关系，不是互相吞并。**

## 文章核心贡献

### 1. Dual memory pathway

文章把标准 SNN 的瞬态膜电位路径和一个低维 slow memory path 分开：

- fast path：保留 spiking / event-driven sparse computation，用于即时证据和快速反应。
- slow path：每层维护一个低维状态 `m in R^d`，其中 `d << N`，通常约为 hidden width 的 5-10%。
- coupling：slow state summarize recent activity，并作为 additional current 调制 fast spiking dynamics。

关键点是 slow path 不是另一个 dense recurrent network，也不是每条边一个 delay buffer，而是 layer-local shared memory。它把长程上下文从 per-synapse / per-neuron 历史中抽出来，压成一个显式状态。

### 2. 长程 credit assignment 的稳定通道

文章用 last-time-step supervision 检查梯度能否回到早期时间步。DMP-SNN 在 S-MNIST / PS-MNIST 上仍能保持有用梯度，而纯 feedforward SNN 会快速衰减。

这说明 slow state 的价值不是“让模型记得更多细节”，而是给长期误差提供一条更稳定的反传路径。对 VZ 的映射是：background-slow / session-medium 不应该只是事后日志或摘要，它们必须能影响 online-fast 的后续可学习状态。

### 3. 算法和硬件共同收敛

文章的硬件侧并不是附属 benchmark，而是和算法结构同构：

- slow state 保持 compact shared state，适合 on-chip / near-memory retention。
- spike integration 是 sparse path；memory integration 是 dense vector-matrix path。
- 硬件用 dependency breaking、operator fusion、heterogeneous operand stationarity 分别优化两条路径。

结果包括：

- 40-60% fewer parameters than comparable SOTA SNNs。
- more than 4x throughput over delay-based digital neuromorphic implementation。
- more than 5x energy efficiency over delay-based designs。

这组数字的领先性来自“结构正确”，不是来自单点算子优化：算法先把状态边界切干净，硬件再镜像这个边界。

## 领先性判断

### 算法领先性

DMP-SNN 解决的是 SNN 里的一个核心张力：长程时间上下文通常依赖 dense recurrence 或 long learnable delays，但这两者都会破坏 SNN 原本的 sparsity / memory efficiency。

它的领先性在于把 temporal memory 从三类昂贵位置移走：

- 不放在 dense recurrent matrix 里，避免 O(N^2) parameter / traffic。
- 不放在 per-connection delay buffer 里，避免深 buffer 和 timing metadata。
- 不放在更大的 neuron time constant 里，避免把所有信息都塞进同一衰减通道。

它选择放在 layer-local low-rank slow state 里。这和 VZ 的 owner 思路一致：长程上下文应该由明确 owner 发布 compact state，下游消费 compact context，而不是重建生产者内部历史。

### 工程领先性

很多神经启发模型停在生物类比层。本文更强的地方是它把 cortical fast-slow motif 降维成一个可以映射到数据流的工程抽象：

- fast sparse path 对应 event-driven spike accumulation。
- slow dense path 对应 compact memory integration。
- 两条路径并行执行，而不是在同一个统一 core 里互相拖慢。

这对 VZ 的启发不是“做硬件”，而是：**能力边界如果在算法层切对，部署层才能切对。** 如果 memory / temporal / reflection 的 owner 边界在 runtime 中含混，后面无论怎么做 orchestration、benchmark 或 deployment 都会被 memory traffic / state reconstruction 的隐性成本吞掉。

### 范式领先性

文章最后的结论是：make the slow state explicit, shared and low-rank; keep the fast path spike-driven; execute the two in parallel.

这句话几乎可以平移成 VZ 的研究原则：

- slow state explicit：memory / reflection / regime / temporal priors 不能藏在 prompt 或 ad hoc local DB 里。
- shared and low-rank：跨 turn / session 的上下文应该是 compact owner snapshot，不是消费者自行拼接所有历史。
- fast path event-driven：online-fast 只做当下必要更新，不承担 durable memory 的第二 owner。
- parallel paths：background-slow 不阻塞 realtime loop，但必须在 context boundary 通过正式 owner path 回流。

## 对 VolvenceZero 的启发

### R1/R5/R6: multi-timescale memory

当前 VZ 的 CMS / memory tower 已经强调 online-fast、session-medium、background-slow 和 rare-heavy 的分层。DMP 进一步提醒两点：

1. slow state 不应只是“低频摘要”，而应是 fast path 的 contextual modulator。
2. slow state 的维度和更新频率应随任务时间结构 co-tune，而不是越大越好。

具体研究方向：

- 在 `continuum_profile.bands[*]` 的未来设计里，把每个 band 的 `state_rank` / `readout_band_id` / `fast_modulation_target` 作为候选 telemetry。
- 在 paper-suite 里增加“slow state ablation”：关闭 slow-to-fast init / fast prior 回灌后，观察 long-horizon recall、family reuse、delayed credit 是否退化。
- 不新建 `DMPMemoryModule`。如果吸收，只能作为 `MemoryModule` owner 内部 tower 演进。

### R3/R4: latent controller and internal control

DMP 的 slow path 不是 standalone recurrent brain，而是调制 fast path 的 additional current。对 temporal owner 来说，这支持一个保守设计：

- `z_t` / `beta_t` 仍是控制器代码和切换门。
- slow memory / reflection fast prior 只作为 owner-side context / bias 进入 switch pressure、family continuation、competition score。
- consumer 不直接读取或更新 slow state 内部结构。

这能避免一个常见错误：把慢记忆升级成“第二个决策器”。DMP 的经验相反：slow path 的价值在于提供 compressed context，而不是取代 fast stimulus-driven computation。

### R8/R15: snapshot mirror of algorithmic separation

本文的 hardware co-design 对 VZ 最深的借鉴是“镜像”：

- 算法有 fast / slow boundary。
- 硬件也有 sparse / dense path boundary。
- 两者共享同一个抽象，因此优化不互相打架。

VZ 的等价要求是：

- runtime owner boundary 要镜像算法 boundary。
- snapshot contract 要镜像 owner boundary。
- rollback / shadow / active wiring 要镜像 state transition boundary。

如果一个能力在 spec 中说是 background-slow，但实现中却在 turn-time 表达层里偷偷拼接历史，那就等价于在 DMP 里把 slow path 又塞回 dense recurrence：短期看能跑，长期必然失去可观测、可回滚和可部署性。

## 不应做的事

- 不把 SNN / neuromorphic hardware 引入近期 runtime 主链。
- 不把 DMP 的 slow state 解释成新的 memory owner。
- 不用页面、prompt 或 BFF route 层模拟 slow state。
- 不把本文 benchmark 数字直接转写成 VZ 性能 claim；它们只支持设计原则，不支持 VZ 已达到类似硬件效率。

## 建议后续行动

### P0: research layer only

- 保持本文在 `research/` 层作为重要外部证据。
- 在后续改 `continuum-memory.md` 或 `multi-timescale-learning.md` 时，可把 DMP 作为“explicit low-rank slow state”的 external evidence。

### P1: proof surface

- 给 memory / temporal paper-suite 增加 slow-path ablation：关闭 fast prior、slow-to-fast reset、tower-native consolidation，检查 long-horizon 行为退化。
- 给 `TemporalAbstractionSnapshot` 的 runtime state 研究一个 compact slow-context influence readout，证明 slow path 只是 modulator，不是第二 owner。

### P2: substrate research

- 把 neuromorphic / SSM / low-rank state path 放入 long-term substrate research pool。
- 只在 future substrate owner 中讨论，不进入 deploy-side BFF 或 product app。

## 结论

DMP-SNN 是一篇值得放入 VZ research 主线的重要文章。它从 SNN + hardware co-design 角度独立证明了一个关键方向：**长期时间能力的正确形态不是更大的统一状态，而是显式、低维、共享、可调制、可镜像部署的慢状态。**

VZ 应吸收的是这个结构原则：fast path 保持在线、稀疏、事件驱动；slow path 保持 compact、owner-local、可审计；两者通过正式 snapshot / owner API 耦合。只要坚持这一点，DMP 对 VZ 的价值就不是硬件路线，而是对“多时间尺度自适应有机体”设计哲学的一次外部强验证。
