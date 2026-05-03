为了弥补从“算法机制”到“真正的伴侣型认知生命”之间的鸿沟，结合我们项目现有的设计（如预测误差、多时间尺度、心智理论、内稳态等），我建议我们在科学端重点深挖以下 **5 个交叉学科理论**：

### 1. 预测编码与主动推断（Predictive Coding & Active Inference）
我们系统的第一性原理是“预测错误（Prediction Error）是学习的唯一原料”。这在认知科学中有一个极具统治力的对应理论：
* **核心代表**：Karl Friston 的**自由能原理（Free Energy Principle）**与主动推断。
* **为什么需要深挖**：传统的强化学习（RL）依赖外部设定的奖励函数，而主动推断认为，生命体的终极目标是“最小化未来的惊奇（Surprise/Prediction Error）”。如果我们能把 Friston 复杂的数学框架，映射到我们 `vz-cognition` 的预测误差追踪中，我们将从根本上解决系统“为何要行动、为何要说话”的内在动机问题，彻底摆脱人工奖励设计。

### 2. 心智理论（Theory of Mind, ToM）的计算模型
我们的文档（`02_theory_of_mind.md`）已经开始把用户的认知拆分为：信念（Belief）、意图（Intent）、感受（Feeling）和偏好（Preference）。但目前的更新规则还相对粗糙。
* **核心代表**：贝叶斯心智理论（Bayesian Theory of Mind, BToM）、递归推理（Recursive Reasoning / 比如“我认为你认为我认为……”）。
* **为什么需要深挖**：高 EQ 的核心在于“懂对方没说出口的话”。我们要研究如何让我们的数字生命不仅能建立单层的用户模型，还能进行反事实推理（Counterfactual Reasoning：“如果我刚才那么说，他会不会生气？”）。这对于实现复杂的信任修复（Rupture & Repair）至关重要。

### 3. 主动学习与学徒学习（Active / Apprenticeship Learning）
在我们的内部宣讲材料（`haiyang_pitch_deck.md`）中，我们将这列为了继 NL 和 ETA 之后的第三个核心理论支柱。
* **为什么需要深挖**：人类不是靠阅读几千亿个 token 变成专家的，而是通过“学徒制”——在交互中遇到不懂的问题，主动发问；在观察专家的行为时，自动提取关键策略。深挖社会学习（Social Learning），研究系统如何高信息熵地**主动向用户索取反馈**，是我们在“数据稀缺的对话场景”中实现高效认知沉淀的关键。

### 4. 记忆巩固的互补学习系统（Complementary Learning Systems, CLS）
我们设计了快（online-fast）、中、慢（background-slow）、极慢（rare-heavy）的学习循环，这与脑科学高度一致。
* **核心代表**：James McClelland 等人提出的海马体-新皮层对话机制（Hippocampal-Neocortical Dialogue）。
* **为什么需要深挖**：我们需要研究人类是如何在晚上睡觉时（离线状态下），把白天的海马体（Episodic Memory / 情景记忆）中的信息回放（Replay），并以此微调新皮层（Neocortex / 基础模型权重）中的概念的。如果能在算法层面完美复现这种“记忆再巩固（Reconsolidation）”，我们的 `rare-heavy` 离线提权路径就能让数字生命真正产生“顿悟”和性格的演进。

### 5. 情感神经科学与内稳态控制（Affective Neuroscience & Homeostasis）
在 `lifeform-vitals.md` 中，我们定义了驱动力（Drive）和稳态（Homeostasis）。
* **核心代表**：Jaak Panksepp 的基础情绪系统（Basic Emotion Systems：如 SEEKING, CARE, PANIC/GRIEF, PLAY 等）、控制论（Cybernetics）。
* **为什么需要深挖**：目前 AI 表现出的“情绪”往往只是文本层面的扮演（Prompt 出来的笑脸）。如果我们深入研究 Panksepp 的情感神经回路以及内稳态的衰减方程，我们可以把“生命体征偏差（Vitals Deviation）”和底层的时序控制器（ETA 里的 $z_t$）挂钩。这样，它的情绪将不再是“文字”，而是因为内部驱动力失衡而自发产生的一种**底层控制策略的倾斜**。

---

**总结**：
我们团队在工程上的 **NL（分层/记忆）** 和 **ETA（抽象/控制）** 已经搭建了极其前沿的“骨架”。接下来的研究重心，应该从纯粹的计算机科学（如把模型再做大），转向**计算认知科学（Computational Cognitive Science）**，把 Friston 的预测编码、McClelland 的记忆系统以及人类心智理论的数学模型“移植”到我们的骨架中，注入真正的血肉才能真正长出来。