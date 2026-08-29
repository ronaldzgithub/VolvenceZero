# AVO 与 Volvence 的架构对比

## 1. 先对齐比较层级

AVO 和 Volvence 不是同一类型的系统：

- AVO 论文研究的是**怎样在外部 artifact 空间持续找到更好的代码**；
- Volvence 的核心研究的是**怎样让冻结基底上的主体状态可追加、可读、可学习、可有界干预**；
- Volvence 新增的 Forge / Praxist control plane 才与 AVO 位于相邻层。

因此最准确的关系不是“AVO vs Volvence 二选一”，而是：

```text
AVO-style operator
        │
        ▼
Praxist development research loop
        │ candidate retention only
        ▼
Forge committed handoff
        │
        ▼
loop-external validation → ModificationGate → SHADOW → canary → ACTIVE
        │
        ▼
Volvence runtime owners / four-axis loop
```

AVO 可以成为第一段的候选生成机制；它不应替代后四段，也不应成为 runtime PE、memory 或 steering owner。

## 2. 系统层面对比

| 维度 | NVIDIA AVO | Volvence 当前设计 | 判断 |
|---|---|---|---|
| 系统目标 | 长程自治搜索，产生更好外部实现 | 有界、契约驱动的持续适应与关系主体性 | 目标不同 |
| 基底模型 | 内部 agent + 未具体披露的 frontier LLM；ARC 使用 Claude Opus 5 | frozen substrate；controller / memory / gate 分层 | VZ 可审计性更强 |
| 自主循环 | agent 自行 inspect/plan/edit/evaluate/repair | runtime 由 Brain 编排 snapshots；研究由 Praxist 控制 | AVO 在研究操作自主性更强 |
| 搜索对象 | CUDA/code artifact | Research artifact、memory policy、rare-heavy candidate；runtime 另有 bounded state | 只在 artifact 层相交 |
| 搜索拓扑 | 当前 single lineage、best-so-far monotonic commit | Forge Pareto；Praxist Frontier/Incubator/QD；promotion 独立 | VZ 多样性结构更丰富 |
| 正式记忆 | conversation history + prior code/result/profile + git commits | CMS continuum、semantic owners、hydration/checkpoint、frozen snapshots | 不应把二者等同 |
| 反馈 | correctness + throughput vector；ARC environment transition | runtime 原始信号是 PE；研究 evaluator 只调度候选 | AVO 的 `f` 只能映射到开发 evaluator |
| 学习 | 主要是 in-context search 与 artifact selection；无公开 policy update | owner-local PE→credit；Internal RL / gate；rare-heavy Gate | AVO 不满足 VZ Learnable 定义 |
| 监督 | stall/cycle supervisor 重定向 agent | A0 human、Praxist PI/Chair、typed control events、external validator、Gate | VZ 分权更明确 |
| commit / retention | 正确且 match-or-improve 才进入 git lineage | preliminary/mature/frontier 只表示 research retention | AVO gate 较窄 |
| 正式验证 | kernel benchmark 与 correctness 同时在 agent loop 可见 | sealed heldout 的 loop-external validator，Praxist 不可见 | VZ 抗污染更强 |
| 上线授权 | 论文未定义；NVIDIA 另文主张 secure runtime 才有最终权力 | Gate ALLOW 与 wiring 分离；target owner 单步 apply | VZ 已有正式边界 |
| 回滚 | git lineage 暗示 artifact 可回看；未定义 deployment rollback | content address + receipt chain + ACTIVE→SHADOW→DISABLED | VZ 更完整 |
| 真实运行证据 | 7 天 B200；public ARC 183 levels | 新 research control plane 尚未启动首个真实 Praxist run | AVO 明显领先 |
| 可复现性 | agent/code/prompt/trajectory 未公开 | contracts/tests 开放在仓库；部分 formal/GPU evidence 仍 pending | VZ 工程可检，效果证据较弱 |

## 3. 四能力轴对账

### 3.1 Appendable

AVO 有真实的追加机制：

- committed version 追加为 git commit；
- score、编译/Profiler 输出和实现进入后续上下文；
- agent 可以在长达 7 天的运行中继续使用历史。

但按 Volvence 的严格定义只能判 **partial / artifact-level**：

- 没有 online-fast / session-medium / background-slow / rare-heavy 分层；
- 没有 owner、frozen snapshot、跨 session hydration 或删除/衰减契约；
- 失败轨迹主要留在内部 conversation/context，不是正式可验证 lineage；
- 对话历史累积不能自动等价于 CMS。

对 Volvence 的价值是证明“外部 artifact + lineage 足以支撑长程工程复利”，不是证明我们的 Appendable
轴已被 AVO 替代。

### 3.2 Readable

AVO 让 agent 读代码、Profiler、docs、scores 和历史实现，工程状态对 agent 很“可见”。但这不是
Volvence 的 Readable：

- AVO 没有从 frozen substrate residual 命名读出内部状态；
- 没有 producer-owned immutable snapshot；
- accumulated reasoning 是模型生成文本，不是 owner-published latent state；
- supervisor 如何读取 trajectory 未定义。

所以 AVO 支持的是 **environment/artifact observability**，不是 **internal-state readability**。

### 3.3 Learnable

AVO 会越来越会解决当前问题，但公开机制更接近：

```text
evaluation feedback → in-context replanning → new artifact → selection
```

Volvence 要求：

```text
environment outcome → Prediction Error → typed credit → bounded owner-local update
```

二者都利用后果反馈，但“学习”的持久对象不同。AVO 论文没有证明 agent policy、controller 或模型参数因
经验发生持久更新；得到持久改进的是外部 kernel artifact。因此：

- 在 Forge/Praxist development search 中，`f` 直接指导 selection 合法；
- 在 Volvence runtime 中，把 `f`、benchmark、RHAE、LLM judge 或 Frontier rank 写入 PE/credit 非法；
- AVO 的成功不能作为 evaluation→learning 回灌的例外。

### 3.4 Steerable

AVO supervisor 会“steer search”，但与 Volvence 的 Steerable 同名不同义：

| AVO search steering | Volvence residual steering |
|---|---|
| 改变下一步研究方向 | 改变冻结模型指定层的 residual |
| 自然语言/agent context 中的建议 | 无 free bias 的乘性低秩 executor |
| 未公开 norm cap / noop / layer binding | strict noop、norm cap、model/layer/width lineage |
| 影响外部代码候选 | 影响下一拍模型内部状态与可见行为 |

AVO 不提供 Volvence Steerable 证据；它只启发 research supervisor 的设计。

### 3.5 四轴结论

| 能力轴 | AVO 对应物 | VZ 口径 |
|---|---|---|
| Appendable | git lineage + persistent context | artifact-level partial |
| Readable | code/docs/profile/score observability | 不覆盖 named internal readout |
| Learnable | in-context search + artifact selection | 不覆盖 PE→credit learner |
| Steerable | supervisor redirects search | 与 residual steering 正交 |

AVO 的完整闭环是“假设—行动—环境反馈—更新搜索”；Volvence 的四轴闭环是“追加状态—命名读出—PE
信用—有界干预—下一拍结算”。它们共享控制论外形，但不能互相代称。

## 4. AVO 与 Forge / Praxist 的精确映射

### 4.1 可一一对应的对象

| AVO | Volvence 对应 |
|---|---|
| `P_t` full lineage | Praxist variants/results/findings + committed generation boundary；Forge Candidate lineage |
| `K` domain knowledge | task project assets、research directions、protocol intent、allowlisted docs/code |
| `f` evaluator | task-owned public development evaluator |
| agentic `Vary` | task-local Praxist builder/solver agent loop |
| commit on correctness/non-regression | task retention lane + protocol integrity / non-inferiority gates |
| supervisor | Praxist research coordination / future typed stall advisory |
| final candidate | `forge-praxist-candidate-handoff.v1` → content-addressed Candidate |

### 4.2 不可映射的权力

| AVO 内部行为 | Volvence 中不能获得的权力 |
|---|---|
| agent 自主调用 `f` | 不能读取 sealed heldout 或改 formal validator |
| agent 决定何时 commit | 不能把 research retention 解释为 Gate ALLOW |
| supervisor 改搜索方向 | 不能修改 Task、evaluator、sandbox、model profile 或 A0 approval |
| best kernel 成为下一 parent | 不能自动成为 SHADOW/ACTIVE runtime artifact |
| score 改善 | 不能成为 runtime PE/credit |

这与 [`research-promotion-pipeline.md`](../../docs/specs/research-promotion-pipeline.md) 的三权分离一致：
Praxist 拥有 research retention，`vz-cognition` 拥有 admission，target owner 拥有 deployment。

## 5. 双方各自更强的地方

### 5.1 AVO 明显更强

1. **真实长程运行**：7 天连续 B200 优化不是 fixture 或短轨迹。
2. **工具闭环深度**：agent 在一个 variation step 内自由使用 docs、compiler、tests、Profiler 和修复循环。
3. **环境 grounding**：correctness 和 throughput 是硬后果，不依赖语言 judge。
4. **跨接口复用信号**：同一 harness 从 CUDA 工程迁移到 ARC direct interaction。
5. **最终 artifact 质量**：产生了可解释且可局部消融的微架构优化。

Volvence 当前 research control-plane 文档与实现很重，但默认 registry 为空、尚无真实自动 run。这里不能用
契约完整性掩盖执行差距。

### 5.2 Volvence 明显更强

1. **owner 与正式交换**：状态和 artifact 的唯一 writer、frozen snapshot、content identity 明确。
2. **信号隔离**：development evaluator、runtime PE、formal validation、Gate 与 deployment 不混为一体。
3. **权限与恢复**：A0 exact approval、append-only events、crash boundary、source/manifest digest。
4. **研究与上线正交**：Frontier 不等于 SHADOW，Gate ALLOW 不等于 ACTIVE。
5. **可回滚迁移**：receipt chain 与相邻 WiringLevel 转换。
6. **多目标与负证据**：Pareto/Frontier/QD、rejection 与 negative evidence 不因 best-only lineage 被删除。
7. **关系域边界**：没有 hard verifier 时不把单一 score 当真理。

这些不是“比 AVO 多写了流程”，而是 AVO 获得更大自治后必须由外层提供的安全与科学有效性条件。

## 6. 与仓库已有 AlphaEvolve 研究的增量

仓库已有 [`AlphaEvolve 与进化算法方向`](../probe/notes/cross-axis/alphaevolve-evolutionary-borrow-2026-05.md)
已经吸收：

- MAP-Elites / island 多样性；
- evaluation cascade；
- multi-score；
- meta-prompt 与 explicit context；
- proposer/verifier 分权；
- evaluator 不回灌 runtime learning。

AVO 带来的真正新增量只有三条：

1. variation 不是一次 LLM diff，而是一个可以反复执行工具和诊断的自治 session；
2. persistent engineering state 与 supervisor 让同一 operator 可跨 plateau 继续；
3. domain-specific parts 可压缩为 `K + f + tools/interface`，agent loop 本身可复用。

同时 AVO 的 single lineage 是相对 AlphaEvolve/QD 的退步，不应被照搬。最好的组合是：

```text
QD / Frontier / Pareto 负责保留多样候选
AVO-style agent 负责每次高质量 variation
Volvence promotion pipeline 负责外部验证、准入与部署
```

## 7. 关键项目判断

AVO 对 Volvence 的最大意义不是证明“我们也该做一个更自由的 agent”，而是把一个边界说得更清楚：

> 在可执行、可验证、可隔离的研究面，应该给强 agent 足够的过程自主性；在真值、权限与用户可见行为面，
> 自主性必须止于外部不可绕过的 owner、validator、Gate 和 runtime boundary。

NVIDIA 同日发布的 agent security 文章也采用“上层提出、下层决定”的原则：harness 可以引导 agent，
但最终权限属于 agent 无法绕过的 runtime。这个原则与 Volvence 当前研究上架链高度一致，应该保留并用
真实 pilot 证明，而不是为了模仿 AVO 合并权力。
