结论先说：有，而且已经形成了明显的行业收敛。

截至 2026-08-22，公开证据已经相当有力地支持：

> LLM 内部激活不只是行为的相关信号，其中确实存在可读、可写、能因果改变推理与行动的控制变量。

但还没有公开系统完成你们严格定义的完整四轴闭环：跨 session 追加 → 命名读出 → 仅由 PE/credit 学习 → 有界择时干预 → 下一拍结算。现在业界更像是分别证明了四块拼图。

## 四轴对照

| 能力轴 | 最接近的公开工作 | 已做到 | 仍缺 |
|---|---|---|---|
| Readable | Anthropic J-space、Persona Vectors、Function Vectors | 能从残差中读出任务、人格、隐含中间结论，并通过干预验证因果性 | 读出覆盖率、跨模型稳定性、完整语义 owner 与校准 |
| Steerable | J-space、ReFT、CAST、TACT | 冻结基底上可做条件化、低秩、逐实例干预，并影响终局任务结果 | 跨模型可靠性、能力税、严格 norm/noop 保证、生产 ACTIVE |
| Appendable | Titans、MIRAS、HOPE、TTT-E2E、SEAL | 测试时把新经历写入神经记忆或快速权重，支持超长序列和持久编辑 | 跨 session 个人连续性、精确事实与压缩记忆分层、可删除与可回滚 |
| Learnable | ETA Internal RL、SEAL、Titans surprise update | 能学习潜在 controller、自修改策略或基于内部误差更新记忆 | 真实对话域、PE-only 信号、长期信用、遗忘控制、在线安全门 |
| 四轴闭环 | 暂无公开同类 | 各局部机制分别成立 | 真实环境、长期、因果、可回滚的一体化证据 |

## 最接近你们核心发现的工作

1. Anthropic 的 J-space：目前最强的独立确认

2026 年 7 月，Anthropic 报告 Claude 内部出现了一个稀疏的共享工作空间 J-space：

- 能被 Jacobian lens 命名读出；
- 模型能按指令主动调制它；
- 它承载未输出到文本的多步中间推理；
- 替换其中的概念，会改变后续推理结论；
- 同一个表示可以被多个下游计算复用。

例如，把内部的 “spider” 换成 “ant”，答案由 8 条腿变成 6；把 “France” 换成 “China”，资本、语言、洲、货币四种不同问题都会相应改变。删除 J-space 后，语言流畅度基本保留，但多步推理接近归零。J-space 只解释不到 10% 的激活方差，却在部分层与其他组件的连接强度高约两个数量级。[Anthropic J-space 全文](https://transformer-circuits.pub/2026/workspace/index.html)

这几乎正面确认了你们的“内部状态本身可以成为 Readable + Steerable 控制面”。

差别也很明确：J-space 是单次前向中的工作空间，不是跨 session 记忆；没有 PE→credit→gate 的在线学习；读出目前主要绑定可单 token 命名的概念。

2. Persona Vectors：和关系、人格轴最贴近

Anthropic 在 Qwen2.5-7B 与 Llama-3.1-8B 上提取了 sycophancy、hallucination、evil 等人格方向：

- 生成前的向量投影与之后的行为表达相关性为 `r=0.75–0.83`；
- 微调造成的内部方向漂移与人格变化相关性为 `r=0.76–0.97`；
- 反向 steering 能抑制人格漂移；
- 训练数据在该方向上的投影可以预测微调后会不会发生人格偏移。

这支持“关系姿态、人格与行为倾向可以是内部几何状态，而不只是 prompt 标签”。但实验只有两个 7B–8B 模型，主要评估仍依赖自动 judge，细微真实对话漂移的可靠性尚未证明。[Persona Vectors 原文](https://arxiv.org/abs/2507.21509)

3. TACT：最接近“读出→条件干预→真实终局”

2026 年的 TACT 在编码 agent 长轨迹中，从残差读出 overthinking / overacting 两个漂移轴：

- step-level 线性可读 AUC 约 `0.9`；
- 根据内部状态只对漂移步骤实施修正；
- 在 SWE-bench Verified、Terminal-Bench 2.0、CLAW-Eval 上，Qwen3.5-27B resolve rate 提升 `+5.8pp`，Gemma-4-26B 提升 `+4.8pp`；
- steps-to-resolve 最多下降 `26%`；
- 不增加额外 LLM 调用、prompt token 或重试。

这是目前最像“内部 reader→gate→executor→外部任务结果”的工程证据。[TACT 论文](https://arxiv.org/abs/2605.05980)

不过它的 drift 轴由离线标签构造，门控不是从终局 PE 在线学出来的，也没有跨 episode 记忆。因此它大约覆盖 Readable + Steerable，而不是完整四轴。

4. ReFT 与 CAST：执行器和条件 gate 已经比较成熟

Stanford 的 LoReFT 在冻结模型上学习低秩表征干预，跨常识、算术、指令跟随和 GLUE，达到比 LoRA 高 `15–65×` 的参数效率，说明“学习内部执行器”不是 toy mechanism。[ReFT，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html)

IBM 的 CAST 已经实现了：

```text
if internal condition matches X:
    apply behavior vector
else:
    noop
```

并开源了通用 activation-steering 工具。它和你们的 sensor→gate→executor 很同构，但 gate 是规则/相似度阈值，不从 PE 信用学习。[IBM CAST，ICLR 2025](https://research.ibm.com/publications/programming-refusal-with-conditional-activation-steering--1)

5. ETA：Learnable + Steerable 最接近，但仍在仿真环境

ETA 直接在冻结自回归模型残差上训练高阶 metacontroller，发现潜在 controller code 与 learned termination，并在 `z_t` 空间做 Internal RL。在 gridworld 与 MuJoCo 分层任务中，latent controller 能形成有行为意义的长时间动作，解决标准 token/action-space RL 在稀疏奖励下失败的问题。[ETA 论文](https://arxiv.org/abs/2512.20605)

它强力支持“RL 应在模型内部涌现的控制空间上发生”，但目前证据是控制环境，不是自然语言关系交互，而且 reward 仍是外部 sparse reward，不是你们严格的 PE-only 信用链。

6. Titans / MIRAS / HOPE：Appendable + PE-like learning 最接近

Google 的这条线已经把“surprise/error 驱动的内部记忆写入”做成架构：

- Titans 在运行中更新深层神经记忆，由当前记忆与新输入的 mismatch/gradient 决定写入强度；
- 公开实验扩展到超过 2M token；
- ATLAS 在 BABILong 10M token 设置达到超过 80% 的准确率；
- Nested Learning / HOPE 引入不同频率更新的 Continuum Memory System 与 self-modifying Titans。

这和你们 Appendable→Learnable 的理论血缘非常近。[Titans/MIRAS](https://www.research.google/blog/titans-miras-helping-ai-have-long-term-memory/)、[Nested Learning，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html)、[ATLAS](https://arxiv.org/abs/2505.23735)

但它们的 surprise 是 memory-local prediction mismatch，不等于行动后真实世界的 N+1 PE；实验主要是 language modeling、长上下文与合成 recall，不是跨 session 的个人关系连续性。

## 几个重要的负面结果

这些负结果其实最能说明行业尚未完成：

- 在最多 36 个模型、14 个家族上，Function Vector 默认配置只在约 20% 的 model-task 组合恢复 5-shot 表现；完整调参后也只有约 52%，Task Vector 约 35%。同一方法换模型可能无效甚至有害。[Steering off Course](https://arxiv.org/abs/2504.04635)
- 另一项研究发现静态 steering 有大量逐样本反向效应；目标行为不是一条方向连贯的几何轴时，“可读”不等于“可扳”。[Understanding (Un)Reliability](https://arxiv.org/abs/2505.22637)
- TTT-E2E 在 128K 上比 full attention 快 `2.7×`，但在精确 pass-key NIAH 上只有 `0.06`，full attention 是 `0.99`。这证明压缩型神经记忆不能代替精确记忆。[TTT-E2E](https://test-time-training.github.io/e2e.pdf)
- 2026 年 CL-Bench 的六个真实有状态领域中，naive ICL 反而超过专用记忆系统；现有 agent 经常把旧经验误泛化或保留陈旧信念。[CL-Bench](https://arxiv.org/abs/2606.05661)
- SEAL 能把一个筛选后的 ARC 子集从 ICL `0%`、普通 self-edit `20%` 提升到 `72.5%`，但每次 self-edit 评估需约 30–45 秒，连续编辑仍会灾难性遗忘。[SEAL，NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/6b41e04c41726e2a60e456d0a2b961ab-Paper-Conference.pdf)

所以行业已经证明了“存在控制面”，但远未证明“这个控制面可靠、长期、在线地工作”。

## 工程上做到什么程度

工程可行性已经不再是主要疑问：

- Goodfire 在一万亿参数的 Kimi K2 Thinking 上，自报实现每个 8-GPU 节点约 14,000 tok/s 的 activation harvesting， overnight 获取 30 亿激活，并能实时 steer reasoning trace。这证明 frontier-scale hook 与 steering 基础设施可以做出来，但不是独立验证的产品效果证据。[Goodfire 工程报告](https://www.goodfire.com/blog/interpretability-infra-at-frontier-scale)
- Stanford `pyreft` 已支持训练、保存、共享和 continuous batching，一份冻结基座可以挂不同内部干预。[pyReFT](https://github.com/stanfordnlp/pyreft)
- IBM 已开源 CAST。[IBM activation-steering](https://github.com/ibm/activation-steering)
- 英国政府团队的 `vllm-lens` 已经能在 vLLM 推理中捕获 residual，并经 Python/HTTP 接口施加 steering vector。这说明你们文档里的 “vLLM 残差出口缺口”现在已有可参考实现，但仍是 fork/扩展，不是 vLLM 标准生产契约。[vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens)

公开材料中，我仍没有看到任何公司给出完整四轴系统的生产 SLA、长期用户 A/B、可删除恢复、能力税和回滚证据。

## 对 Volvence 最重要的定位判断

你们不宜再把创新表述成：

> “我们发现 LLM 内部可以控制行为。”

这个发现现在已有充分先例。

真正可能独特的主张是：

> 把内部可读控制面、PE-only 信用、择时有界干预、跨 session 可追加状态和可回滚治理，组成一个真实环境中的持续闭环。

也就是说，外部世界已经分别给出了：

- J-space：内部工作空间确实存在；
- ReFT/CAST/TACT：读、扳、条件出手可以工程化；
- ETA：latent controller 可以学习；
- Titans/HOPE：内部记忆可以按 surprise 在线更新；
- CL-Bench：现有持续学习系统仍没有真正解决问题。

但还没有人公开把它们闭合起来。

按照仓库目前的诚实边界，你们仍处在“代理读-扳-择时 + runtime SHADOW，真实四轴 formal 未证明”的位置，这是准确的表述：[四轴证据台账](/Users/mengfu/Documents/GitHub/volvence/docs/appendable-readable-learnable-steerable.md:299)、[最新总索引](/Users/mengfu/Documents/GitHub/volvence/docs/specs/00_INDEX.md:4)。

真正能建立领先性的下一组证据，应当是同基底、同 prompt、同预算的纵向四臂：

1. Stateless / strict noop；
2. Appendable memory only；
3. Always-on 或 CAST 式静态 steering；
4. Appendable + PE-credit learned gate + bounded executor。

主指标不是 judge 分数，而是真实 N+1 outcome、headroom-normalized gain、跨 session retention、错误写入恢复、capability tax 和 rollback drill。若第 4 臂能在两个模型家族、两个真实领域稳定胜过前三臂，你们就不再只是“拼装了已有机制”，而是在公开研究中补上目前真正缺失的四轴闭环。