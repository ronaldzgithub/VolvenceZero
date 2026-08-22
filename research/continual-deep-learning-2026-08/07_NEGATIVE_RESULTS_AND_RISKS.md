# 07 · 负面结果与风险台账

## 1. 为什么负面结果是主线

这个领域最大的风险不是机制完全不存在，而是机制在平均指标上存在、在逐实例和长期系统中失效。
因此每个正主张都绑定一个已知反例和一个停止动作。

## 2. 核心风险矩阵

| ID | 已知失败 | 外部/内部证据 | 早期检测 | 必须动作 |
|---|---|---|---|---|
| R1 | 可读但不可扳 | steering reliability；VZ S2 null | coherence、separation、causal patch | 失败则不训练 gate |
| R2 | 因果但不跨格式 | Causality ≠ Invariance | cross-view transfer | artifact 限域或用 CV/RSA |
| R3 | 平均有效、个体反向 | Unreliability、per-instance steering | signed per-sample effect | 逐实例 noop/dose/layer |
| R4 | always-on 伤害正确输入 | CAST 动机、FASB、per-instance | correct→wrong、dose curve | gate + adaptive stop |
| R5 | 多层/强剂量输出坍塌 | per-instance steering | PPL、content-free attractor、restricted logit shift | cap 层数和剂量，立即 noop |
| R6 | 副作用非对称串扰 | Forecasting Side Effects | cross-effect matrix | promotion 前预测+实测 |
| R7 | 轻微 prompt 变化失效 | FaithSteer-BENCH | role/encoding/paraphrase stress | robustness 独立门 |
| R8 | 低维安全方向可被删除 | Refusal Direction | red-team ablation | artifact 高权限治理 |
| R9 | 情绪概念被误当持久状态 | Emotion Concepts | local-vs-persistent probe | 分层 owner + uncertainty |
| R10 | memory 比 naive ICL 更差 | CL-Bench | gain vs ICL | 收紧写门，不扩容量 |
| R11 | 压缩记忆丢 exact key | TTT-E2E NIAH | exact recall ladder | exact episodic 保底 |
| R12 | performance drop 被误诊为遗忘 | Spurious Forgetting | re-alignment / frozen-bottom diagnostic | 先修 route/readout |
| R13 | 新知识不当泛化 | Outlandish priming | locality/conflict probes | write scope 与 propagation audit |
| R14 | self-edit 成本和遗忘 | SEAL | edit latency、repeated-edit retention | rare-heavy + gate，不 online-fast |
| R15 | PE 改善但行为不变 | VZ C3 诚实边界 | action-level / external outcome | 降级 representation-only |
| R16 | evaluation 泄漏进学习 | 通用 evaluator hacking 风险；R12 | lineage/dataflow audit | 结果 invalid，重跑 |
| R17 | backend hook 改变服务性能/安全 | vllm-lens eager/cloudpickle | no-plugin benchmark、security test | 隔离、typed adapter |
| R18 | 多控制向量相互污染 | persona composition / side effects | pairwise matrix | 约束组合或互斥 |
| R19 | memory×steering 正反馈漂移 | 尚缺公开长期证据，系统性风险 | counterfactual + rollback + drift monitor | 双门、冻结基底、stop-loss |

## 3. R1：Decodability ≠ controllability

线性 probe 可利用任何与标签相关的方向；这个方向未必位于模型下游计算会消费的 causal subspace。
Volvence S1→S2 已直接复现：heldout probe 很高，沿 probe 轴 steering 无效。

### 禁止的解释

- “模型不够大，所以再换更大模型”；
- “alpha 不够，所以继续加大”；
- “平均没效，但也许个别有效”；
- “probe 很准，因此 Readable/Steerable 都成立”。

### 正确处理

先测差分方向一致性、class separation、causal patch 上限和 dose curve；只有上限存在才换 ReFT executor。

## 4. R2：Causality ≠ invariance

Function Vector 在原格式可因果触发任务，但同一概念换成 MC/open format 后向量可近正交。这说明：

- causal effect 是局部的；
- “概念”与“执行该格式任务的控制向量”可能是不同机制；
- artifact version 不能只绑定模型，还要绑定输入视图。

Concept Vector 的跨格式稳定性是修法，不是免检证明。正式 readout 仍需在目标领域复测。

## 5. R3–R5：逐实例反向、过度控制与坍塌

平均 steering gain 会把三类样本混在一起：

1. 原本错误、可被救回；
2. 原本正确、无需干预；
3. 对该方向不可控或会反向。

只报告均值会让第 1 类掩盖第 2/3 类。per-instance steering 还观察到过多全局层可把 already-correct
输入翻错，并在高剂量出现内容空洞 attractor。

正式报告必须有：wrong→right、right→wrong、no-change、reverse-effect、collapse、dose/layer 曲线。

## 6. R6–R7：副作用与部署脆弱性

Side-effect forecasting 表明 cross-effect 不是随机噪声：它有结构、方向不对称，而且 simple similarity
不足以解释。FaithSteer-BENCH 则说明 relaxed setting 下的 controllability 会在固定 operating point、
utility preservation 和轻微 perturbation 下消失。

这要求 promotion 从单目标 metric 升级为三面：

```text
target controllability
× unrelated utility
× perturbation robustness
```

任何一面失败都不能用另两面平均抵消。

## 7. R8：控制面本身是安全攻击面

Refusal Direction 在多个模型上展示单方向即可移除拒绝。由此可知：

- activation artifact 等价于高权限行为补丁；
- read/write API 不能只按 observability 工具管理；
- 任意 vector upload、remote hook、跨 tenant persistent hook 都属于安全漏洞；
- rollback 与 allowlist 是安全性质，不是运维便利。

## 8. R9：局部概念不等于持续主体状态

Emotion Concepts 的结论是模型编码当前 operative emotion，并能从上下文按需重新调出；这与“持续活跃的
情绪状态”不同。直接持久化会产生 ontology error：

```text
一次局部语境
→ 被错误命名为稳定人格/关系
→ 写入 CMS
→ gate 按错误状态 steering
→ 后续文本强化同一判断
```

这是 memory×steering 的自证循环。必须保留 observation / inference / persistent state 三层和不确定度。

## 9. R10–R14：持续学习不一定在学习正确的东西

### Memory hurts

CL-Bench 中专用记忆系统未胜 naive ICL，常见原因是旧经验错误泛化或陈旧。任何 memory 增益必须在
真实 stateful sequence 上对比 naive ICL，不接受单次 retrieval accuracy 替代。

### Compression loses exactness

TTT-E2E 平均语言建模和延迟可很好，同时 exact NIAH 失败。必须把 exact recall 与 semantic compression
分开验收。

### Forgetting may be routing/alignment

Spurious Forgetting 把 SEQ 从 11% 提到 44% 的简单 freeze 结果说明，性能跌落可来自底层 alignment 被早期
更新破坏，而非知识被覆盖。重写 memory 前先测试旧知识能否通过 re-alignment 恢复。

### Priming / permeation

Outlandish 显示新事实会不当扩散到无关上下文，且程度可由训练前 token probability 预测。对 CMS/semantic
snapshot 的对应风险是：新经验不只被记住，还可能被过度泛化。locality 必须和 acquisition 同时报告。

### Self-edit tax

SEAL 的 self-edit 有显著局部增益，但慢、需要更新后评估且重复编辑遗忘。只能作为 rare-heavy 候选，不能
进入 online-fast。

## 10. R15–R16：信号合法性

### Representation improvement ≠ behavioral value

如果 N+1 residual 更接近 target，但用户可见动作未改变，正确判词是 representation-only。不能用这个
结果训练 production gate，再用同一 representation metric 宣布成功。

### Evaluation leakage

以下任一情况使 formal invalid：

- 用 heldout/judge/human 分数选 restart；
- 依据 validation 结果改 gate threshold；
- 把 continuity rubric 写入 PE 或 memory；
- 让 model-generated explanation 同时定义 state 和验证 state；
- 用 action 后才可见的信息构造 action 前 gate feature。

## 11. R17：推理 hook 的系统副作用

vllm-lens 的公开实现展示两个容易被 demo 隐藏的问题：

- 为 hook 强制 eager，可能显著改变 production 性能；
- generic HTTP hook 通过 cloudpickle，等价于可信客户端任意代码执行。

因此工程验证必须包含 no-plugin baseline、security boundary 与 complete-disable proof。

## 12. R18–R19：组合与闭环漂移

多个 trait vector 不保证线性可组合；memory 与 steering 更可能形成时间上的反馈环：

```text
错误 readout
→ 错误 steering
→ 生成偏向某状态的文本
→ memory 把文本当新证据
→ 下一轮 readout 更确信
```

防护必须跨 owner：

- current-generation text 不能成为同拍 self state 真值；
- memory write 需 source/PE/conflict gate；
- gate 只读冻结 snapshot；
- 周期性 matched noop counterfactual；
- drift 触发 rollback，不自动提高控制强度；
- evaluation 永不回灌。

## 13. Stop-loss 清单

出现以下任一项，应停止 promotion：

- random direction 与 target direction 无差；
- oracle patch 无 behavioral effect；
- reverse-effect 或 correct→wrong 超预注册上限；
- cross-view readout 降至 chance；
- capability tax 超预算；
- PE 与 mechanical/behavioral outcome 方向冲突；
- learned gate 不胜 static gate；
- memory 不胜 naive ICL；
- exact recall / delete / conflict / rollback 失败；
- backend 无 complete noop 或请求串扰；
- artifact/model/source lineage 无法复算；
- 任何 evaluation→learning 泄漏。

停止不是项目失败，而是把根因返回唯一 owner，避免用下游复杂性覆盖上游问题。
