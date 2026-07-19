# 核心论文作者延伸阅读与成熟度评估

调研日期：2026-05-06

核心论文来源：

- `2512.24695` — Nested Learning: The Illusion of Deep Learning Architectures
- `2512.20605` — Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning

收录口径：

- NL 线以 Ali Behrouz / Meisam Razaviyayn / Peilin Zhong / Vahab Mirrokni 为主。
- ETA 线以 Seijin Kobayashi / Johannes von Oswald / Alexander Meulemans / João Sacramento 等主要技术作者为主。
- 只收录与 VolvenceZero 的 NL/ETA、连续记忆、test-time learning、latent/internal control、信用分配、组合泛化、多主体关系学习有直接价值的论文。
- 已排除明显偏图挖掘、时间序列应用、蛋白模型、信息抽取等弱相关论文。

成熟度标尺：

- 高：顶会/期刊已接收，或理论与实验链条完整，可直接进入 spec / benchmark 设计。
- 中：arXiv 新作但实验面较完整，适合进入技术路线或 shadow prototype。
- 低：立意重要但仍偏 proposal / workshop / 单点实验，适合观察，不宜直接改运行时主链。

## 一、总体结论

最值得立刻吸收的是三组：

1. **NL 记忆主线**：`Titans`、`ATLAS`、`Miras`、`TNT`、`Memory Caching`、`Trellis`。它们把 NL 主论文里的“优化器即记忆 / test-time memorization / continuum memory”落成了更工程化的结构。对 `vz-memory`、`vz-substrate` 的长上下文、记忆分层、快慢记忆接口都有直接参考价值。
2. **ETA / 内部学习主线**：`Uncovering mesa-optimization`、`MesaNet`、`From Growing to Looping`、`Depth-Grown Models`。它们共同说明：模型内部可以形成可识别的优化过程、循环计算和 test-time computation。对 `vz-temporal` 的 metacontroller、internal RL、β/z 抽象有支撑意义。
3. **信用与关系学习主线**：`COCOA`、`Least-control principle`、`Deep Feedback Control`、`Learning-aware policy gradients`、`In-context co-player inference`。这组成熟度最高，对 `vz-cognition/credit`、delayed attribution、relationship_state、multi-agent self/other modeling 的价值大于直接代码移植。

不建议立刻吸收为运行时核心的是 `Embedded Universal Predictive Intelligence`：它很贴合“embedded agency / self-prediction / infinite-order ToM”的方向，但目前是 203 页理论框架，成熟度偏低，适合放进长期研究路线。

## 二、NL 作者线

### 1. Titans: Learning to Memorize at Test Time (`2501.00663`)

文件：`research/papers/titans-learning-to-memorize-at-test-time-2501.00663.pdf`

- 作者线索：Ali Behrouz / Peilin Zhong / Vahab Mirrokni。
- 核心价值：把 attention 视为短期记忆，把神经记忆模块视为更持久的 long-term memory，直接对应 VolvenceZero 的“记忆连续谱”。
- 对我们意义：可作为 `vz-memory` 四层记忆与 `vz-substrate` 长上下文能力的工程参照。尤其是“可训练神经记忆 + attention 当前上下文”的组合，很接近 transient/episodic/persistent 的运行时分层。
- 成熟度：中高。实验面覆盖语言建模、common-sense、genomics、time series、needle-in-haystack，工程可读性强；但仍是新近 arXiv，需验证开源实现与复现实验。
- 建议：高优先级精读。不要直接引入模型结构，先抽象为 CMS 的 memory update / retrieval / retention 设计原则。

### 2. It's All Connected / Miras (`2504.13173`)

文件：`research/papers/miras-test-time-memorization-attentional-bias-2504.13173.pdf`

- 作者线索：NL 四位核心作者全员。
- 核心价值：把 Transformers、Titans、现代线性 RNN 统一解释为 associative memory modules，并把 forget gate 解释为 retention regularization。
- 对我们意义：这是 `docs/next_gen_emogpt.md` 里“优化器/架构/记忆是一类东西”的最好桥梁论文。适合补强 `vz-memory` 的 retention、forgetting、memory learning algorithm 术语。
- 成熟度：中。概念统一性很强，实验横跨多个模型族；但新框架较大，工程落地需要拆小。
- 建议：进入架构术语表。可把“attentional bias objective / retention gate / memory learning algorithm”映射到 CMS slot 描述，而不是新增跨模块状态。

### 3. ATLAS: Learning to Optimally Memorize the Context at Test Time (`2505.23735`)

文件：`research/papers/atlas-optimally-memorize-context-test-time-2505.23735.pdf`

- 作者线索：Ali Behrouz、Peilin Zhong、Meisam Razaviyayn、Vahab Mirrokni。
- 核心价值：强调固定大小 recurrent memory 的三类不足：容量受限、只对最近 token 在线更新、记忆管理不够表达。
- 对我们意义：直接提醒 `MemoryStore` / ReflectionEngine 不能只做追加式卡片，也要有 owner 内部的压缩、重写、retention policy。对 long-context / episodic-to-semantic consolidation 很有价值。
- 成熟度：中。实验声称在长上下文和 BABILong 上明显提升，但仍需看复现性。
- 建议：高优先级。适合转化为 `vz-memory` 的 memory-management acceptance：容量、压缩、过去上下文再优化，而非简单召回。

### 4. TNT: Improving Chunkwise Training for Test-Time Memorization (`2511.07343`)

文件：`research/papers/tnt-chunkwise-training-test-time-memorization-2511.07343.pdf`

- 作者线索：Ali Behrouz、Peilin Zhong、Meisam Razaviyayn、Vahab Mirrokni。
- 核心价值：用“全局粗粒度记忆 + 多个局部高分辨率记忆”的两阶段训练解决 test-time memorization 的训练效率问题。
- 对我们意义：非常贴近 NL 多时间尺度。可给 `online-fast / session-medium / background-slow` 的训练与评估提供类比：先粗粒度稳定，再局部精修。
- 成熟度：中。工程目标明确，有速度收益；但偏训练流程，不是运行时模块。
- 建议：用于 rare-heavy / offline refresh 的 pipeline 设计，不要进入在线 turn path。

### 5. Trellis: Learning to Compress Key-Value Memory in Attention Models (`2512.23852`)

文件：`research/papers/trellis-compress-kv-memory-attention-models-2512.23852.pdf`

- 作者线索：Ali Behrouz / Vahab Mirrokni。
- 核心价值：固定大小 KV memory + test-time recurrent compression + forget gate，COLM 2025。
- 对我们意义：对 `vz-substrate` 的 residual/KV cache steering 与 bounded adapter-delta 很有启发：控制入口必须有容量边界和压缩机制。
- 成熟度：高于同批 NL 新作。已有 COLM 2025 发表记录，实验较工程化。
- 建议：作为 bounded memory / bounded control 的成熟参考，可进入 substrate spec 的参考文献。

### 6. MS-SSM: Multi-Scale State Space Model (`2512.23824`)

文件：`research/papers/ms-ssm-multi-scale-state-space-model-2512.23824.pdf`

- 作者线索：Ali Behrouz / Peilin Zhong / Vahab Mirrokni。
- 核心价值：多尺度 SSM，用不同 resolution 捕获细粒度与全局依赖，COLM 2025。
- 对我们意义：理论上对多时间尺度有启发，但主要是 backbone architecture，不应直接影响运行时 owner 边界。
- 成熟度：高于普通 arXiv，COLM 2025；但与当前产品主链的距离中等。
- 建议：中优先级。作为 `vz-substrate` 后续 open-weight backbone 调研，不进入 `vz-runtime`。

### 7. Memory Caching: RNNs with Growing Memory (`2602.24281`)

文件：`research/papers/memory-caching-rnns-growing-memory-2602.24281.pdf`

- 作者线索：NL 四位核心作者均参与。
- 核心价值：给 RNN 增加可增长的 memory checkpoint cache，在固定内存 RNN 与 Transformer 全上下文之间做复杂度/召回折中。
- 对我们意义：对 `vz-memory` 的 snapshot replay、session checkpoint、episodic cache 设计有直接启发：不要只保存最终摘要，关键状态 checkpoint 也应可作为可验证证据。
- 成熟度：中偏低。2026 新作，思路清晰但还需等待外部验证。
- 建议：用于评估“checkpoint 作为记忆 stratum”的设计，不急于实现。

### 8. CoDE-Stop: Confidence Dynamics Early Stopping (`2604.04930`)

文件：`research/papers/code-stop-confidence-dynamics-early-stopping-2604.04930.pdf`

- 作者线索：Meisam Razaviyayn。
- 核心价值：用中间答案 confidence dynamics 判断 reasoning 是否应提前停止，减少过度推理。
- 对我们意义：可作为 expression / reasoning budget 的 readout，而不是决策主逻辑。它提醒我们：推理深度也应是可观测动态，而非固定 prompt 策略。
- 成熟度：中偏低。新作，方法简单，复现成本可能低。
- 建议：适合进入 evaluation/report-only，暂不进入控制器在线决策。

## 三、ETA 作者线

### 9. Uncovering mesa-optimization algorithms in Transformers (`2309.05858`)

文件：`research/papers/mesa-optimization-algorithms-in-transformers-2309.05858.pdf`

- 作者线索：Johannes von Oswald / Alexander Meulemans / Seijin Kobayashi / João Sacramento。
- 核心价值：证明 next-token prediction 可诱发模型内部的 subsidiary learning algorithm，解释 in-context learning 何以像优化。
- 对我们意义：这是 ETA 的前置理论支柱之一。它支撑 VolvenceZero “token 之上存在内部控制/学习空间”的设计，而不是把所有适应写成 prompt。
- 成熟度：中高。2023 工作，影响面较大，实验是 synthetic 但机制清晰。
- 建议：高优先级精读。可作为 `vz-temporal` metacontroller 的理论依据。

### 10. MesaNet: Sequence Modeling by Locally Optimal Test-Time Training (`2506.05233`)

文件：`research/papers/mesanet-locally-optimal-test-time-training-2506.05233.pdf`

- 作者线索：Johannes von Oswald / Seijin Kobayashi / Alexander Meulemans / João Sacramento。
- 核心价值：把 recurrent layer dynamics 解释为 in-context objective，并用局部最优 test-time training 提升长上下文能力。
- 对我们意义：和 NL 线形成交叉：内部优化不是 metaphor，而可以是 layer-level computation。适合指导 `vz-temporal` 的 SSL pretraining target 与 open-weight residual evidence。
- 成熟度：中。大规模实验更接近工程，但仍是较新 arXiv。
- 建议：高优先级。适合放入 ETA paper-suite 的 stronger mechanism evidence 背景。

### 11. From Growing to Looping: Iterative Computation in LLMs (`2602.16490`)

文件：`research/papers/growing-to-looping-iterative-computation-llms-2602.16490.pdf`

- 作者线索：Johannes von Oswald。
- 核心价值：把 depth growing 与 inference-time looping 统一为 iterative computation 的不同表现。
- 对我们意义：对 `vz-temporal` 的“循环使用控制器 / β_t 切换 / more compute when needed”有启发。也支持不要把“更多 token”误等同于“更多思考”。
- 成熟度：中偏低。2026 新作，实证方向重要但需等待外部引用。
- 建议：观察。可用于 reasoning budget / controller reuse 的设计讨论。

### 12. Do Depth-Grown Models Overcome the Curse of Depth? (`2512.08819`)

文件：`research/papers/depth-grown-models-curse-of-depth-2512.08819.pdf`

- 作者线索：Johannes von Oswald。
- 核心价值：分析 depth-grown Transformer 如何改变 residual stream 结构、形成更有效的计算块。
- 对我们意义：支撑“有效层/有效模块”不是固定架构给定，而会随训练形成。这对 open-weight residual-control evidence 很重要。
- 成熟度：中偏低。新作，机制分析有价值，但离产品实现较远。
- 建议：作为 substrate / residual analysis 背景，不作为当前工程优先项。

### 13. Discovering modular solutions that generalize compositionally (`2312.15001`)

文件：`research/papers/modular-solutions-compositional-generalization-2312.15001.pdf`

- 作者线索：Seijin Kobayashi / Johannes von Oswald / João Sacramento。
- 核心价值：研究何时可以发现可组合模块，并证明/实证模块识别与组合泛化的条件。ICLR 2024，带代码。
- 对我们意义：直接对应 ETA 的 abstract-action family 与 held-out composition gate。它说明可组合性需要结构条件，不是“多训练一些”自然保证。
- 成熟度：高。ICLR 2024 + code，是本批最成熟参考之一。
- 建议：立刻吸收进 ETA strong-proof benchmark 的 held-out composition 解释与失败模式。

### 14. When can transformers compositionally generalize in-context? (`2407.12275`)

文件：`research/papers/transformers-compositionally-generalize-in-context-2407.12275.pdf`

- 作者线索：Seijin Kobayashi / Johannes von Oswald / João Sacramento。
- 核心价值：指出 Transformer in-context composition 常失败，只有引入显式瓶颈以分离 task inference 与 task execution 时才更可靠。
- 对我们意义：这非常重要：VolvenceZero 的 `z_t / β_t` 不应只是隐式希望模型学会，而应通过 owner/snapshot 边界和 latent bottleneck 明确隔离。
- 成熟度：中。ICML workshop，机制结论强，但规模较小。
- 建议：纳入 `vz-temporal` 设计原则：抽象动作瓶颈是必要结构，不是可选优化。

## 四、信用、控制与关系学习

### 15. COCOA: Counterfactual Contribution Analysis (`2306.16803`)

文件：`research/papers/cocoa-counterfactual-contribution-credit-assignment-2306.16803.pdf`

- 作者线索：Alexander Meulemans / Seijin Kobayashi。
- 核心价值：用反事实问题“如果采取别的动作，还会得到这个 reward 吗？”改进长期信用分配。NeurIPS 2023 spotlight。
- 对我们意义：几乎正中 `vz-cognition/credit` 的 delayed attribution。尤其适合关系/repair 场景：不要只看结果好坏，要估计某个 turn/regime/action family 对结果的贡献。
- 成熟度：高。顶会 spotlight，理论与实验都强。
- 建议：最高优先级。应进入 credit spec，并影响 `claim_rare_heavy_net_benefit`、delayed outcome attribution 的评估口径。

### 16. The least-control principle for local learning at equilibrium (`2207.01332`)

文件：`research/papers/least-control-principle-local-learning-equilibrium-2207.01332.pdf`

- 作者线索：Alexander Meulemans / Seijin Kobayashi / Johannes von Oswald / João Sacramento。
- 核心价值：把学习表述为减少系统达到目标状态所需的控制量。NeurIPS 2022。
- 对我们意义：对 `ModificationGate` 和 `vz-cognition/credit` 很有启发：好的适应不是更强干预，而是让系统更少依赖外部/上层控制。
- 成熟度：高。NeurIPS 2022，理论成熟。
- 建议：用于 gate/evaluation 的哲学与指标补强，例如把“控制量下降”作为 report-only evidence。

### 17. Credit Assignment in Neural Networks through Deep Feedback Control (`2106.07887`)

文件：`research/papers/deep-feedback-control-credit-assignment-2106.07887.pdf`

- 作者线索：Alexander Meulemans / João Sacramento。
- 核心价值：反馈控制信号可用于信用分配，并近似 Gauss-Newton 优化。
- 对我们意义：给“控制信号也是信用信号”的设计提供老而稳的理论背景。可帮助解释为什么 prediction_error / credit / controller update 应走同一主链，而不是 evaluation 直接写学习。
- 成熟度：高。2021 论文，理论与补充材料完整。
- 建议：作为 credit/control 基础参考，不需要直接实现。

### 18. Multi-agent cooperation through learning-aware policy gradients (`2410.18636`)

文件：`research/papers/learning-aware-policy-gradients-multi-agent-cooperation-2410.18636.pdf`

- 作者线索：Alexander Meulemans / Seijin Kobayashi / Johannes von Oswald / João Sacramento。
- 核心价值：agent 显式考虑其他 agent 的学习动态，可在 social dilemmas 中形成合作。
- 对我们意义：对 relationship_state、user_model、dual_track 很重要。真实伴侣场景里，用户不是静态环境，用户也在适应系统。
- 成熟度：中高。实验和理论解释较完整，但仍偏多智能体 benchmark。
- 建议：用于 open-dialogue benchmark 的长期关系评价，不直接写入响应策略。

### 19. Multi-agent cooperation through in-context co-player inference (`2602.16301`)

文件：`research/papers/in-context-co-player-inference-multi-agent-cooperation-2602.16301.pdf`

- 作者线索：João Sacramento / Alexander Meulemans 参与。
- 核心价值：sequence model 可在 episode 内通过 in-context learning 推断 co-player，而不需硬编码对方学习规则。
- 对我们意义：支持 user_model 走“结构化 proposal + owner-owned update”，不要写关键词规则或固定 persona router。
- 成熟度：中偏低。2026 新作，规模较小但方向正。
- 建议：观察并用于 relationship/user_model 研究，不进入主链。

### 20. Embedded Universal Predictive Intelligence (`2511.22226`)

文件：`research/papers/embedded-universal-predictive-intelligence-2511.22226.pdf`

- 作者线索：Alexander Meulemans / Seijin Kobayashi / João Sacramento 等。
- 核心价值：从 embedded agency 与 self-prediction 出发，给多主体学习和自我建模一个统一理论框架。
- 对我们意义：非常贴合 VolvenceZero 的双轨、自我轨道、relationship_state、user_model 方向。但它更像理论蓝图，不是工程方法。
- 成熟度：低到中。203 页，理论性强，暂缺可直接落地的工程 acceptance。
- 建议：长期研究参考。不要用它替换当前 snapshot/owner 边界。

## 五、当前优先级

第一档，建议本周精读并转化为 spec / benchmark 语言：

1. `2306.16803` COCOA
2. `2312.15001` Discovering modular solutions
3. `2501.00663` Titans
4. `2504.13173` Miras
5. `2309.05858` Uncovering mesa-optimization
6. `2506.05233` MesaNet

第二档，建议作为设计备份和 shadow prototype 输入：

1. `2505.23735` ATLAS
2. `2511.07343` TNT
3. `2512.23852` Trellis
4. `2602.24281` Memory Caching
5. `2207.01332` Least-control principle
6. `2106.07887` Deep Feedback Control

第三档，继续观察，不进入近期实现：

1. `2511.22226` Embedded Universal Predictive Intelligence
2. `2602.16301` In-context co-player inference
3. `2602.16490` From Growing to Looping
4. `2512.08819` Depth-Grown Models
5. `2604.04930` CoDE-Stop
6. `2512.23824` MS-SSM

## 六、对 VolvenceZero 的具体落点

- `vz-memory`：吸收 Titans / ATLAS / Miras / Memory Caching 的“记忆不是单一 store，而是带 retention、compression、checkpoint、re-optimization 的连续谱”。
- `vz-temporal`：吸收 mesa-optimization / MesaNet / compositional bottleneck 的证据，把 `z_t / β_t` 继续保持为显式 latent/controller 层，不退化成 prompt 标签。
- `vz-cognition/credit`：吸收 COCOA、least-control、DFC，把 delayed attribution 做成“贡献估计”，不是结果分数回填。
- `vz-evaluation`：把成熟论文中的 evidence style 转为 report-only metrics：composition generalization、counterfactual contribution、control effort reduction、memory checkpoint utility。
- `vz-runtime`：不新增跨模块直接调用。上述论文只应影响 owner 内部 snapshot enrichment 或 evaluation readout，不能破坏 snapshot SSOT。

## 七、陪伴向优先级附录（2026-05-06，含 NL/ETA 对齐性裁判）

> 本附录已经过一次自我修正。第一版把 A-Mem 和 HippoRAG 2 列为"立竿见影候选"，
> 这是错的：它们与 NL/ETA 的核心命题"记忆即架构、更新规则被学到、检索在 forward
> computation 里"直接冲突。CMS / Continuum Memory 已在实现中，再叠加 A-Mem 等于
> 给 memory owner 树立第二个由外部 LLM 当 curator 的影子 owner，会让两条记忆
> 策略漂离。下表是修正后的版本。

NL/ETA 三条裁判线（任何"记忆向"候选必须同时通过）：

1. 不创造与 CMS 并列的第二个 memory owner；演化只发生在 CMS / artifact owner 内部。
2. 更新规则不被外置 LLM curator 替代；记忆写入与组织由系统自身 owner 与学习信号驱动。
3. 检索不引入与 substrate 表征并行的独立嵌入空间；retrieval 应当是 substrate / CMS readout 的下游使用方。

修正后的陪伴向优先级：

| 排序 | 论文 | 落点 | 与 NL/ETA 的关系 | 时间窗 |
|---|---|---|---|---|
| 1 | `2306.16803` COCOA | `vz-cognition/credit` 内部算法升级 | 给 ETA Internal RL 提供低方差信用估计，作者同一脉 | 本周 |
| 2 | `2604.18701` Curiosity-Critic | `vz-cognition/prediction` 内部 critic readout | 让 NL 多频率写入收到更干净的 PE 信号；不创造并行结构 | 本周–下周 |
| 3 | `2502.14802` HippoRAG 2 的**算法片段**（多跳 / 图召回） | 仅作为 `DerivedRetrievalIndex` 的 owner-internal 召回算法可选项 | NL-中性。前提是 embedding 已切到 substrate feature；不把 PageRank 图当成第二 memory owner | 在第 5 项之后 |
| 4 | `2501.00663` Titans + `2505.23735` ATLAS 的 **memory update rule** 部分 | CMS 内部的 update rule 升级（不是替换基底 LLM） | NL 作者线对陪伴最对齐的部分；它是 memory 模块层面的工作，可以在当前 substrate 不变下做 | 中期专项 |
| 5 | retrieval embedding 切到 substrate feature | `memory/retrieval.py::_semantic_embedding` 替换为 `SubstrateSnapshot.feature_signals` 下游使用 | 这条本身不是论文，是 NL 对齐补丁；解除当前 hash placeholder 现状 | 本周–下周 |

明确**不进路线图**：

- `2502.12110` **A-Mem**：让外部 LLM 当 memory curator，违反 NL 第 2 条裁判线（更新规则被学到）。CMS / Hope 已经是 NL-原生答案，A-Mem 是同问题的非 NL 解。
- 把 HippoRAG 2 当作完整 memory 架构上：违反 NL 第 1 条与第 3 条（并行 owner + 并行嵌入空间）。仅保留它的局部召回算法作为 owner-internal 候选。

陪伴向**降级**的（不要按"NL/ETA 作者线必须紧跟"这条逻辑投资到本季度路线）：

- Titans / ATLAS 的**架构主体部分** / Miras / TNT / Trellis / MS-SSM：作为完整架构换装本季度无感。注意：`Titans` / `ATLAS` 的"memory update rule"部分被单独提到上表第 4 行，这是它们在陪伴向**真正有价值**的子集。
- Uncovering mesa-optimization / MesaNet / Growing-to-Looping / Depth-Grown：设计原则参考，工程兑现远。
- Embedded UPI、Multi-agent learning-aware、In-context co-player inference：理论或多智能体 benchmark，单用户陪伴近期不需要。

陪伴向改动的不变量（在前面三条裁判线之外）：

- 所有改动只在对应 owner 内部丰富快照；消费者不重建生产者状态（R8）。
- 不引入关键词/规则映射；新读出值必须从嵌入/critic 等学习方法产出（`no-keyword-matching-hacks`）。
- 不允许 evaluation 或 LLM 评分回写 credit / regime / memory；它们仍是 PE 的下游 readout（`real-open-dialogue-learning-loop.md` 的红线）。
- 改动必须可回滚（owner 内部 checkpoint），与 `ModificationGate` 协议一致。

## 八、Substrate-level uplift 的成本梯子（2026-05-06）

> 这一节是 NL/ETA 论文落到我们仓时的**成本/可行性裁判表**。以后讨论"是否要把 X 论文落到代码"时，先把它定位到下面四档之一，不再笼统说"substrate 替换"。

### Tier 3 —— 冻结基底 LLM

`vz-substrate` 里 `OpenWeightResidualStreamSubstrateAdapter` 包装的 HF 模型本身。R2 写死：永不再训。**任何要求改这一层的论文，本季度不做。**

### Tier 2 —— substrate / memory 上的可学控制层（已存在的 Hope 脚手架）

物理位置：

- `SubstrateDeltaAdapterLayer` + `OpenWeightResidualInterventionBackend`（HF forward pass 上挂 hook + 学到的 delta，走 `SubstrateOnlineFastCheckpoint` / `SubstrateRareHeavyCheckpoint`）。
- `SubstrateSelfModModule` + `SubstrateFastMemoryCell`（每层 residual 一个 fast memory cell，带 momentum / error_trace / write_gate）。
- `CMSMemoryCore` 三 band 2 层 residual MLP + `LearnedUpdateRule`（已经有 `_hope_*` 字段；nested 变体已实现 NL Appendix A.5 的"慢 band meta-learn 快 band 初始化"）。

提升方式：改更新方程的 Python 代码 + 走现有 cadences 重训本层权重；基底 LLM 不动。

代表论文：**Titans / ATLAS / Hope** 的 memory update rule 部分（不是它们的整体架构）；本仓的 `cms-atlas-titans-uplift` 计划属于此档。

### Tier 1 —— 在 owner 内部新增小型学习头

不动任何已有模块的 weights，只在 owner 内挂一个 critic / contribution model，从 runtime 数据在线训练。

代表：**Curiosity-Critic** 在 `vz-cognition/prediction` 内加 epistemic vs aleatoric critic。**COCOA 的可选 learned contribution model** 也是这一档。

### Tier 0 —— 纯算法变更，无 weights

只读上游快照，写出更好的派生量。**完全不训练**。

代表：

- 把 `memory/retrieval.py::_semantic_embedding` 从 hash 占位换成消费 `SubstrateSnapshot.feature_signals`。
- COCOA 的封闭形式 contribution baseline。
- KV cache steering 的 one-shot intervention。

### 当前候选的 tier 映射

| 候选 | Tier | 是否需要训练 | 工程风险 |
|---|---|---|---|
| `2306.16803` COCOA（基础） | 0 | 否 | 低 |
| `2306.16803` COCOA（learned contribution） | 1 | 是（小头在线） | 低 |
| `2604.18701` Curiosity-Critic | 1 | 是（小 critic 在线） | 低 |
| `_semantic_embedding` 切到 substrate 特征 | 0 | 否 | 极低 |
| `2501.00663` Titans + `2505.23735` ATLAS 的 update rule 进 CMS | 2 | 是（重训 CMS band MLP，**基底 LLM 不动**） | 中 |
| Titans / ATLAS 整体架构 | 3 | 是（基底替换） | 不做 |

### 结论

陪伴向"立竿见影"的真正候选都集中在 Tier 0–2。Tier 3 一律不在本季度路线。**之前把"NL 作者线 = substrate 替换"的判断只在 Tier 3 意义上成立**，把它们的 memory update rule 部分搬进 CMS 是 Tier 2，可做。

## 九、已下载 PDF

新增 PDF 均位于 `research/papers/`：

- `titans-learning-to-memorize-at-test-time-2501.00663.pdf`
- `miras-test-time-memorization-attentional-bias-2504.13173.pdf`
- `atlas-optimally-memorize-context-test-time-2505.23735.pdf`
- `tnt-chunkwise-training-test-time-memorization-2511.07343.pdf`
- `trellis-compress-kv-memory-attention-models-2512.23852.pdf`
- `ms-ssm-multi-scale-state-space-model-2512.23824.pdf`
- `memory-caching-rnns-growing-memory-2602.24281.pdf`
- `code-stop-confidence-dynamics-early-stopping-2604.04930.pdf`
- `mesa-optimization-algorithms-in-transformers-2309.05858.pdf`
- `modular-solutions-compositional-generalization-2312.15001.pdf`
- `transformers-compositionally-generalize-in-context-2407.12275.pdf`
- `learning-aware-policy-gradients-multi-agent-cooperation-2410.18636.pdf`
- `mesanet-locally-optimal-test-time-training-2506.05233.pdf`
- `embedded-universal-predictive-intelligence-2511.22226.pdf`
- `in-context-co-player-inference-multi-agent-cooperation-2602.16301.pdf`
- `growing-to-looping-iterative-computation-llms-2602.16490.pdf`
- `depth-grown-models-curse-of-depth-2512.08819.pdf`
- `cocoa-counterfactual-contribution-credit-assignment-2306.16803.pdf`
- `least-control-principle-local-learning-equilibrium-2207.01332.pdf`
- `deep-feedback-control-credit-assignment-2106.07887.pdf`
