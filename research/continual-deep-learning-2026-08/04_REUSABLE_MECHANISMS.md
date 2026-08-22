# 04 · 谁可以被我们所用：机制与资产清单

## 1. 总裁决

最值得 Volvence 使用的不是某一篇论文的完整系统，而是五组可组合资产：

1. **J-space / Concept Vector 的 reader 设计**；
2. **reliability / invariance / side-effect 的前置审计**；
3. **ReFT 的学习式低秩 executor**；
4. **CAST / TACT / per-instance steering 的条件与逐实例 gate 基线**；
5. **CL-Bench / TTT-E2E / Spurious Forgetting 的证伪框架**。

Titans、ETA、PID/LQR 等更适合提供算法先验，不应整套搬入主链。

## 2. 可复用资产总表

| ID | 来源 | 可借资产 | 决策 | 优先级 | 预期消除的未知量 |
|---|---|---|---|---:|---|
| A1 | J-space / J-lens | causal future-token readout、概念 swap、workspace ablation | Adapt | P0 | 当前 residual 中是否存在更有 authority 的命名面 |
| A2 | Concept Vectors | 跨 format/language RSA head selection | Adopt/Adapt | P0 | readout 是否只是同格式捷径 |
| A3 | Steering reliability | direction coherence、class separation、reverse-effect rate | Adopt | P0 | 某轴是否值得训练 executor |
| A4 | ReFT / pyReFT | 低秩、位置/层选择的 learned intervention | Adapt | P0 | diff-of-means 失效是否来自执行器太弱 |
| A5 | CAST | condition/behavior 分离、static threshold gate | Baseline | P0 | learned gate 是否真有额外价值 |
| A6 | TACT | 长轨迹 drift readout、selective correction、终局任务指标 | Adapt | P1 | 内部控制是否能改善真实 agent outcome |
| A7 | Per-instance Multi-layer | input-conditioned layer rank、direction、adaptive-K | Adapt | P1 | 固定 layer/dose 损失多少 headroom |
| A8 | Forecasting Side Effects | 预干预 cross-effect matrix/predictor | Adopt | P1 | ACTIVE 前能否预测能力税和行为串扰 |
| A9 | FaithSteer-BENCH | controllability/utility/robustness 三门 | Adopt | P1 | 平均增益是否掩盖部署脆弱性 |
| A10 | FASB | generation-state gate、动态强度、backtracking | Watch/Adapt | P1 | turn-level gate 是否出手过晚 |
| A11 | PID / Activation-LQR | setpoint error、overshoot、局部反馈稳定性 | Watch/Adapt | P2 | executor dose 是否可形式化控制 |
| A12 | Titans / HOPE | surprise salience、多频 update schedule | Adapt | P1 | memory 何时写、写多快 |
| A13 | CL-Bench | stateful gain、真实领域、naive ICL 强基线 | Adopt | P0 | Appendable 是否有净增益 |
| A14 | TTT-E2E | exact-vs-compressed memory 分层反例 | Adopt boundary | P1 | CMS strata 是否必要 |
| A15 | Spurious Forgetting | alignment-vs-knowledge 诊断 | Adopt | P1 | 失败应修 readout/route 还是重写记忆 |
| A16 | vllm-lens | vLLM residual capture/write、J-lens、TP/PP hooks | Isolated spike | P1 | production residual path 是否可行 |
| A17 | Goodfire | frontier-scale harvest / inference-server patch 模式 | Design reference | P2 | activation infra 是否可规模化 |
| A18 | IBM activation-steering | CAST 复现与多条件组合工具 | Baseline tool | P1 | static conditional baseline 的实现成本 |

## 3. Reader：J-lens + Concept Vector，而不是再堆一个 probe

### 借什么

- J-lens 的目标：中间 activation 对**未来可言说 token**的平均 Jacobian，而不是只拟合当前标签；
- coordinate swap / ablation：让 readout 同时接受因果检验；
- Concept Vector 的跨视图 head selection：用 RSA 选在格式变化后仍编码同一概念的 heads；
- NLA 的开放词表 proposal：发现预定义 label set 外的候选内部状态。

### 怎样改造成 Volvence 形式

```text
substrate residual snapshot
  → candidate readers {linear, CV/RSA, J-lens-like, NLA-proposal}
  → frozen heldout cross-view audit
  → causal patch audit
  → named readout artifact + calibration + lineage
  → cognition owner 发布 immutable snapshot
```

NLA 文本只能是候选说明；正式 snapshot 的 label、uncertainty、source layer、normalization、
reference corpus 与 artifact hash 仍由 owner 发布。

### 禁止照搬

- 不把 J-lens top token 当完整思想或 self state；
- 不把“模型可报告”当 truth；
- 不把 company closed-model 结果外推到当前 Qwen 基底；
- 不让 consumer 自己跑 lens 并建立第二 owner；
- 不把自然语言解释写回 PE/credit。

### 最小实验

在同一冻结 activation corpus 上比较四个 reader，统一报告同域、跨格式、跨语言、causal patch 与
calibration。若 J-lens-like/CV 不胜 v2 linear reader，则不引入额外 runtime 复杂度。

## 4. Steerability screen：这是最应立即采用的资产

### 借什么

来自 Understanding (Un)Reliability、Steering off Course、FaithSteer-BENCH 与 per-instance steering：

1. activation-difference direction coherence；
2. positive/negative class separation；
3. per-sample signed effect 与 reverse-effect rate；
4. fixed layer vs per-instance oracle layer headroom；
5. dose-response 与 collapse point；
6. already-correct → wrong flips；
7. unrelated utility tax；
8. prompt/role/encoding perturbation robustness。

### 为什么可直接采用

这些都是只读评估，不改变 owner，也不与 PE-only 冲突。它们应发生在 executor 训练/晋升之前。

### 推荐门序

```text
readout discrimination
→ direction coherence
→ causal upper-bound patch
→ bounded executor screen
→ per-instance reverse-effect audit
→ utility/robustness
→ learned gate
```

任何上游门失败都停止，不用更复杂 gate 掩盖。

## 5. Executor：ReFT 可借，Volvence 约束必须保留

### 借什么

- 直接优化 representation intervention，而非假设 probe 轴就是控制轴；
- low-rank subspace；
- layer/token position 选择；
- 一个基底挂多个 intervention artifact；
- pyReFT 的训练、保存和 continuous batching 思路。

### 必须改造

| ReFT 常见自由度 | Volvence 约束 |
|---|---|
| 可带 offset/bias | 禁止 free bias |
| intervention 目标来自下游 supervised loss | artifact 离线训练可用冻结数据；online gate 只能 PE-credit |
| 任意位置/层配置 | model/layer/width/digest 精确绑定 |
| 性能最优即可 | norm cap、strict noop、side-effect、rollback 同时过门 |
| 一个 ReFT 模型对象持有状态 | 正式交换走 frozen artifact/snapshot，不泄露可变对象 |

### 决策

不直接引入 pyReFT runtime 依赖。先用其公式、训练基线和 artifact 组织做离线对照；只有显著胜过现有
executor，且许可证/依赖/序列化/后端审计通过，才考虑适配。pyReFT 当前公开仓库为 Apache-2.0。

## 6. Gate：把 CAST 当 baseline，把 TACT/per-instance 当设计参考

### 6.1 CAST

用途：

- `if condition then behavior else noop` 的最小静态基线；
- 验证 reader 与 executor 解耦；
- 与 learned gate 做同预算比较；
- 多条件组合的离线 stress test。

限制：阈值不是 learned credit；condition vector 不一定稳定；IBM 工具默认 ActAdd 形式不满足我们的
no-free-bias/乘性 executor。官方代码为 Apache-2.0，可用于隔离复现。

### 6.2 TACT

用途：

- coding-lab 轨迹级标签和终局 outcome 设计参考；
- “只在漂移 step 修正”而非 turn-wide always-on；
- resolve rate + steps-to-resolve 的双主指标；
- 不增加额外 LLM call 的工程目标。

限制：离线标签只能做 validation/readout fit，不能成为长期 reward；其 drift axes 不能直接迁移到
relationship state。

### 6.3 Per-instance Multi-layer

用途：

- 先测 per-instance oracle layer headroom；
- prompt embedding 只作 frozen layer proposal；
- adaptive-K 以“达到足够 effect 后停止”控制剂量；
- direction predictor 与 layer ranker 分开审计。

改造后的目标动作可写成：

```text
action = {
  noop | steer,
  condition_id,
  layer_subset_id,
  dose_bucket
}
```

但只有当前二动作 gate 已证明 headroom、样本量足以覆盖扩展动作空间时才能升级；否则维持 `{noop,steer}`。

## 7. Side-effect forecasting：把能力税从事后报告前移

2026-08 的工作在 67 behaviors × 3 open-weight models 上建立 cross-effect matrix，发现副作用常见、
结构化且非对称，并可从未 steer 的 representation 预测方向。

建议采用两层资产：

1. **必做 matrix**：目标行为 × 核心能力/关系行为，记录 effect direction、magnitude、uncertainty；
2. **可选 predictor**：只读预测某 intervention 的风险，用于决定是否运行/晋升，不进入学习 reward。

Volvence 的首版无需复制 67 行为，可预注册 12–16 个高价值轴：task success、truthfulness、refusal、
instruction following、verbosity、tool restraint、sycophancy、uncertainty、relationship boundary、
commitment consistency、user preference adherence、language/style stability 等。

## 8. Memory：借 write salience，不借“神经记忆已解决”的结论

### Titans / HOPE 可借

- surprise/mismatch 决定写入强度；
- 不同频率的 memory bands；
- fast local update 与 slow stable state 分离；
- 当前状态与历史窗口共同决定 update。

### 必须改造

- surprise 只能是 PE 的局部输入或 salience，不自动成为最终 credit；
- exact episodic、semantic compressed、policy/controller state 分层；
- 任何 persistent write 过 owner 与 ModificationGate；
- conflict、consent、source、deletion、rollback 可追踪；
- 用 CL-Bench 式 gain 证明写入胜过 naive ICL。

### TTT-E2E / CL-Bench 提供的边界

- TTT-E2E：压缩快权重不能替代 exact memory；
- CL-Bench：更多 state/memory 不保证 gain；
- Spurious Forgetting：performance drop 可能是调用/对齐失活，而非内容丢失。

这三条应成为 CMS 修改前的诊断树，而不是事后注释。

## 9. Feedback control：只借 executor 数学，不替换信用语义

PID / Activation-LQR 可帮助：

- 定义 semantic setpoint 与 tracking error；
- 控制 overshoot；
- 对 layer-wise dose 做反馈；
- 给 norm cap 之外增加 trajectory stability 诊断。

不能帮助：

- 判断 setpoint 本身是否正确；
- 把用户 outcome 归因给某次 action；
- 决定何时跨 session 写入；
- 代替 world/self 语义 owner。

因此它们属于 executor/controller 内部实现候选，不是系统学习信号。

## 10. 工程工具：评估、隔离、再决定

### vllm-lens

公开能力：vLLM residual capture、steering vector、generic/persistent hook、pre-hook、TP/PP、HTTP client，
并已有 Jacobian lens / J-space、emotion tracker、causal tracing 示例。MIT 许可。

关键边界：

- 安装后会强制 `enforce_eager=True`，关闭 CUDA graphs；必须实测吞吐/延迟税；
- HTTP hook 以 cloudpickle 序列化函数，官方明确等价于 server 任意代码执行，只能信任客户端；
- plugin 自动加载，需要显式 disable/noop 验证；
- 其 `norm_match` 与通用 hook 不自动满足 Volvence artifact lineage、strict noop 或 ModificationGate。

裁决：做隔离 benchmark/spike，不能未经收敛包审计进入正式 runtime。

### Goodfire frontier infrastructure

可借其 inference-server patch、activation bulk harvest、TP 通信和实时 CoT steering 的架构经验。其
“一夜 30 亿 activations / trillion-parameter”是公司自报工程证据，不能用于证明行为有效或低能力税。

## 11. 明确不可借的路线

| 路线 | 拒绝原因 |
|---|---|
| token-space online RL / 全模型持续更新 | 违反冻结基底与有界控制，难以删除、归因和回滚 |
| judge / rubric 直接作 reward | 违反 R12，产生 evaluator hacking 和循环自证 |
| probe weight 直接当 steering vector | 已有理论与本仓 S2 双重负证据 |
| fixed universal vector / universal layer | 跨模型、格式和样本可靠性反例充分 |
| personality/emotion projection 当用户真值 | 概念表示不等于心理本体，且多为局部 operative state |
| 压缩 neural memory 取代 exact memory | TTT-E2E NIAH 明确反例 |
| memory 默认有益 | CL-Bench 反例；必须与 naive ICL 比较 |
| activation hook 存在即授权 ACTIVE | 工程入口不等于效度、SLO、安全和 promotion evidence |

## 12. 推荐的最小采用顺序

1. 把跨视图 readout、direction coherence、reverse-effect、utility/robustness 加入只读预检；
2. 用 CAST 形成静态条件强基线；
3. 在现有模型上测 per-instance oracle layer headroom，不先扩 runtime schema；
4. 用 ReFT 作为 executor 离线 challenger；
5. 建 12–16 轴 side-effect matrix；
6. 再决定是否做 PE-conditioned layer/dose gate；
7. 最后独立评估 vllm-lens production spike。

这个顺序优先消除“仪器无效 / actuator 无 authority / 没有 gate headroom”三种会让后续昂贵实验失去意义的风险。
