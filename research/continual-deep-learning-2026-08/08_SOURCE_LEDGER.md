# 08 · 一手来源与证据台账

## 1. 读法

- `V1`：同行评审会议主会；`V2`：arXiv / 实验室 research thread；`V3`：官方工程报告或代码。
- `M1–M5` 见 [`00_RESEARCH_CHARTER.md`](00_RESEARCH_CHARTER.md)。
- “本包用途”是 Volvence 研究裁决，不代表作者自己的主张。
- 本地已有 PDF 时复用，不重复下载；网页型 research thread 保留官方链接。

## 2. 内部表征、读取与因果控制

| ID | 来源 | 状态 | 可核验主张 | 本包用途 | 本地/相邻材料 |
|---|---|---|---|---|---|
| S01 | [Function Vectors in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html) | ICLR 2024 · V1 · M2 | 少量 heads 搬运紧凑任务表示；注入可在新上下文触发任务 | Readable+Steerable 基础证据 | web |
| S02 | [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html) | NeurIPS 2024 · V1 · M2 | 13 个开源 chat models、最高 72B 的拒绝可由单方向增删 | 低维因果控制 + 安全攻击面 | web |
| S03 | [ReFT: Representation Finetuning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html) | NeurIPS 2024 · V1 · M2 | 冻结基底上学习低秩 intervention；报告比 LoRA 高 15–65× 参数效率 | learned executor | [`PDF`](../steering-2026-08/papers/reft-representation-finetuning-loreft-2404.03592.pdf) |
| S04 | [Persona Vectors](https://arxiv.org/abs/2507.21509) | arXiv 2025 · V2 · M2 | trait direction 可监控、预测和缓解 fine-tuning 后人格漂移 | relationship/persona 邻接证据 | web |
| S05 | [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) | Transformer Circuits 2026 · V2 · M2 | 171 个 emotion concepts 形成可解释几何并因果影响偏好/行为；主要是 local operative concept | relationship readout + 持久化边界 | [`专项`](../anthropic-emotion-concepts-2026-04/README.md) |
| S06 | [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html) | Transformer Circuits 2026 · V2 · M2 | 用自然语言瓶颈重建 activation，解释随训练更有信息 | open-vocabulary readout proposal | web |
| S07 | [Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html) | Transformer Circuits 2026 · V2 · M2 / IC-2 | J-space 可报告、调制、承载内部推理和跨函数复用；swap/ablation 有因果效应 | 最强内生工作空间证据 | web |
| S08 | [Causality ≠ Invariance](https://proceedings.iclr.cc/paper_files/paper/2026/hash/86b3697c4eb7792c951831636bfdacd5-Abstract-Conference.html) | ICLR 2026 · V1 · M2 | causal Function Vectors 跨输入格式可近正交；Concept Vectors 跨格式/语言更稳 | readout lineage 与 RSA 修法 | web |

## 3. 条件、动态、逐实例与反馈 steering

| ID | 来源 | 状态 | 可核验主张 | 本包用途 | 本地/相邻材料 |
|---|---|---|---|---|---|
| S09 | [Programming Refusal with Conditional Activation Steering](https://research.ibm.com/publications/programming-refusal-with-conditional-activation-steering--1) | ICLR 2025 · V1 · M3 | 依据 prompt activation 条件选择是否施加 behavior vector | static-gate 强基线 | [`PDF`](../steering-2026-08/papers/conditional-activation-steering-cast-2409.05907.pdf) |
| S10 | [Steering When Necessary: FASB](https://papers.neurips.cc/paper_files/paper/2025/hash/0c6c92a0c5237761168eafd4549f1584-Abstract-Conference.html) | NeurIPS 2025 · V1 · M3 | 生成中动态判断必要性/强度并 backtrack 修正 | token/trajectory gate 设计 | web |
| S11 | [Activation Steering with a Feedback Controller](https://arxiv.org/abs/2510.04309) | arXiv 2025 · V2 · M3 | 把常见 steering 解释为 P control，引入 PID 跨层反馈 | overshoot / stability 先验 | web |
| S12 | [TACT](https://arxiv.org/abs/2605.05980) | arXiv 2026 · V2 · M3 | coding-agent drift AUC≈0.9；两模型 resolve rate +5.8pp/+4.8pp，steps 最多 -26% | 真实 agent 终局模板 | web |
| S13 | [Local Linearity … Activation-LQR](https://arxiv.org/abs/2604.19018) | arXiv 2026 · V2 · M3 | 用 layer Jacobian/LQR 做 closed-loop semantic setpoint tracking，并分析 tracking error | 反馈 executor 候选 | web |
| S14 | [Deployable Per-Instance Multi-Layer Activation Steering](https://arxiv.org/abs/2608.08829) | arXiv 2026 · V2 · M3 | 最佳层逐实例变化；部署配方回收约 93%/65% oracle lift，降低全局层 damage | layer/direction/dose scheduler | web |
| S15 | [Forecasting Side Effects of Activation Steering](https://arxiv.org/abs/2608.11227) | arXiv 2026 · V2 · M3 audit | 67 behaviors × 3 models；副作用常见、结构化、非对称且可预判 | cross-effect / capability-tax gate | web |

## 4. Steering 可靠性与反证

| ID | 来源 | 状态 | 可核验主张 | 本包用途 | 本地/相邻材料 |
|---|---|---|---|---|---|
| S16 | [Understanding (Un)Reliability of Steering Vectors](https://arxiv.org/abs/2505.22637) | arXiv / workshop 2025 · V2 · M2 audit | 样本级常出现反向 effect；direction coherence 与 class separation 预测成功 | steerability precheck | [`PDF`](../steering-2026-08/papers/understanding-unreliability-of-steering-vectors-2505.22637.pdf) |
| S17 | [Steering off Course](https://aclanthology.org/2025.acl-long.974/) | ACL 2025 · V1 · M2 audit | 最多 36 models / 14 families 中 DoLa/FV/TV 效果高度不稳定，常退化 | 禁止 universal artifact | [`PDF`](../steering-2026-08/papers/steering-off-course-reliability-challenges-acl2025.pdf) |
| S18 | [FaithSteer-BENCH](https://arxiv.org/abs/2603.18329) | arXiv 2026 · V2 · M3 audit | 固定部署 operating point 下暴露虚假可控、能力税和轻微扰动脆弱性 | controllability/utility/robustness 三门 | [`PDF`](../steering-2026-08/papers/faithsteer-bench-deployment-stress-test-2603.18329.pdf) |
| S19 | [From Weights to Activations](https://aclanthology.org/2026.acl-long.1377/) | ACL 2026 · V1 · survey | 表征工程分类、适用范围、可靠性与规模化问题 | 分类学，不单独承载因果结论 | [`PDF`](../steering-2026-08/papers/from-weights-to-activations-repe-survey-acl2026-long1377.pdf) |

## 5. 持续记忆、测试时学习与 Internal RL

| ID | 来源 | 状态 | 可核验主张 | 本包用途 | 本地/相邻材料 |
|---|---|---|---|---|---|
| S20 | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) | arXiv 2025 · V2 · M3 | surprise 驱动 neural memory；报告可扩至 >2M context | memory write salience | [`PDF`](../papers/titans-learning-to-memorize-at-test-time-2501.00663.pdf) |
| S21 | [ATLAS](https://arxiv.org/abs/2505.23735) | arXiv 2025 · V2 · M3 | 高容量、历史感知 memory；BABILong 10M context 报告 >80% | history-aware write / capacity | web |
| S22 | [Nested Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html) | NeurIPS 2025 · V1 · M3 | 多层/并行 context flow、deep optimizer、self-modifying Titans、CMS/HOPE | 多时间尺度理论血缘 | [`PDF`](../papers/nested-learning-illusion-of-deep-architectures-2512.24695.pdf) |
| S23 | [End-to-End Test-Time Training for Long Context](https://test-time-training.github.io/e2e.pdf) | arXiv 2025 · V2 · M3 | 3B/164B；128K latency 比 full attention 快 2.7×；exact NIAH 反例 | fast-weight 价值与 exact-memory 边界 | [`PDF`](../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf) |
| S24 | [Self-Adapting Language Models / SEAL](https://papers.neurips.cc/paper_files/paper/2025/hash/6b41e04c41726e2a60e456d0a2b961ab-Abstract-Conference.html) | NeurIPS 2025 · V1 · M4（慢） | 模型生成 self-edit 与更新指令；下游 reward 训练编辑策略 | rare-heavy 参照 / R2,R12 反例 | [`PDF`](../papers/continual-learning-2607/seal-self-adapting-language-models-2506.10943.pdf) |
| S25 | [Emergent Temporal Abstractions](https://arxiv.org/abs/2512.20605) | arXiv 2025 · V2 · M4（仿真） | latent controller、learned termination、Internal RL 在 sparse reward 分层任务有效 | gate/termination 近邻 | web |
| S26 | [Continual Learning Bench](https://arxiv.org/abs/2606.05661) | arXiv 2026 · V2 · M3 eval | 六个 expert-validated stateful domains；naive ICL 胜专用 memory systems | longitudinal gain 与强基线 | [`PDF`](../papers/continual-learning-2607/continual-learning-bench-stateful-environments-2606.05661.pdf) |
| S27 | [Spurious Forgetting](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html) | ICLR 2025 · V1 · M2 diagnosis | 性能跌落常是 task alignment；freeze bottom layers 将 SEQ 11%→44%，其他方法最高 22% | alignment-vs-knowledge 诊断 | [`PDF`](../papers/continual-learning-2607/spurious-forgetting-continual-learning-lm-2501.13453.pdf) |
| S28 | [How new data permeates LLM knowledge and how to dilute it](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0f85efb1e7545dc35a1b5e4d45aaf3c2-Abstract-Conference.html) | ICLR 2025 · V1 · M2 | 新事实可不当 priming 无关上下文；两种方法降低不良扩散 50–95% | memory locality / propagation 风险 | web |

## 6. 工程实现与基础设施

| ID | 来源 | 状态 | 可核验主张 | 本包用途 | 关键边界 |
|---|---|---|---|---|---|
| S29 | [pyReFT](https://github.com/stanfordnlp/pyreft) | 官方代码 · V3 | ReFT 训练/保存/分享、单基底多 intervention、continuous batching；Apache-2.0 | executor challenger | 不自带 Volvence owner/norm/PE contract |
| S30 | [IBM activation-steering](https://github.com/IBM/activation-steering) | 官方代码 · V3 | ActAdd/CAST、多条件规则与复现实例；Apache-2.0 | static-gate baseline | 默认 additive 与离线 threshold |
| S31 | [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) | 官方代码 · V3 | vLLM residual capture/write、TP/PP、HTTP、J-lens/causal tracing；MIT | production-path spike | 强制 eager；cloudpickle hook 仅可信客户端 |
| S32 | [Goodfire: Interpretability Infrastructure at Frontier Scale](https://www.goodfire.com/blog/interpretability-infra-at-frontier-scale) | 公司工程报告 2026 · V3 | 自报 trillion-param Kimi K2 上 overnight harvest 3B activations 与实时 CoT steer | 可扩展性先验 | 非独立 efficacy / SLA 证据 |

## 7. 相邻仓库证据

| 材料 | 作用 |
|---|---|
| [`../steering-2026-08/README.md`](../steering-2026-08/README.md) | VZ S2 null、C2 executor、S3-E gate 的完整本地机制链 |
| [`../continual-learning-2026-07/README.md`](../continual-learning-2026-07/README.md) | CL-Bench、Spurious Forgetting、SEAL 与 agent memory 横扫 |
| [`../ttt-e2e-long-context-2026-08/README.md`](../ttt-e2e-long-context-2026-08/README.md) | TTT-E2E 定量深读与 exact-memory 裁决 |
| [`../anthropic-emotion-concepts-2026-04/README.md`](../anthropic-emotion-concepts-2026-04/README.md) | emotion vectors、功能边界和 Volvence 映射 |
| [`../../docs/appendable-readable-learnable-steerable.md`](../../docs/appendable-readable-learnable-steerable.md) | 四轴正式定义与当前证据台账 |
| [`../../docs/specs/steering-runtime.md`](../../docs/specs/steering-runtime.md) | reader/gate/executor owner、C1/C3/B3 与 ACTIVE 边界 |

## 8. 来源空白

本轮未找到公开一手材料同时报告：

- 跨 session persistent state + internal reader + learned activation gate；
- learning signal 严格来自 action-conditioned PE，而非 label/judge/reward；
- memory-only、static steering、full-loop 的 matched longitudinal factorial；
- user-visible behavior、deletion、capability tax、SLO 和 rollback 同时通过；
- 完整四轴 production SLA 或长期真实用户 A/B。

这是一项检索结论，不是“私有行业系统必然不存在”的断言。
