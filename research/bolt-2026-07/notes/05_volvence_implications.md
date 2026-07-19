# 对 Volvence 的启发与证伪检查

> 核心问题：BOLT 主创与相邻文献是否支持、修正或证伪 Volvence / EmoGPT 的现有方案？

## 0. 总判断

这批文献没有证伪 Volvence 的核心路线，反而加强了几个关键不变量：

- 在线适应应在有界 latent / controller / owner-local state 中完成，而不是在线改全量基底。
- Transformer 可以被训练成摊还贝叶斯推断器，不只是语言生成器。
- 固定容量 latent memory 对 online-fast 有价值，但不足以替代多时间尺度记忆和多 owner contract。
- Bayesian 名义必须落实为 prior、evidence、posterior、uncertainty、递归误差，而不能只是一段“会更新的 hidden state”。

真正被挑战的是 Volvence 当前实现/文档的一个薄点：我们已经有 R-PE、CMS、ETA、owner snapshots，但对 `owner-local Bayesian update kernel` 的形式化还不够。BOLT 方向提示我们应补一个 SHADOW 级候选机制，而不是改掉架构。

## 1. 可吸收的机制

### 1.1 AmortisedBeliefUpdater

建议定义一个 owner 内部算子，而不是新 runtime owner：

```text
OwnerPrivatePosterior_t
OwnerSnapshot_t
TypedEvidence_t
PredictionError_t
    -> OwnerPrivatePosterior_t_plus_1
    -> OwnerSnapshot_t_plus_1
```

约束：

- updater 不跨 owner 共享 hidden state。
- updater 不直接消费 raw text；只消费 typed evidence / PE。
- public output 仍由 owner 发布 immutable snapshot。
- 默认 SHADOW，与现有 owner update 并跑比较。

适合候选：

- `user_model` 的偏好强度与置信度。
- `relationship_state` 的信任 / rupture / repair 趋势。
- `world_temporal` 的任务环境 belief。
- `self_temporal` 的关系 regime / repair strategy belief。
- `prediction_error` 的 uncertainty readout。

### 1.2 Fixed-capacity latent memory 的定位

固定容量 latent memory 应定位为：

```text
online-fast owner-private posterior cache
```

不应定位为：

```text
全局长期记忆 / 关系事实唯一来源 / 跨模块共享状态
```

原因：

- Continuous Latent Contexts 证明少量 latent token 能承载在线算法状态，但任务是合成在线学习，不是多年关系事实。
- Palimpsa 证明固定容量记忆必须处理 stability-plasticity，不可能无成本替代 replay / consolidation。
- Distribution Transformers 证明递归 posterior 有用，但递归误差会累积，需要慢层校准。

### 1.3 Uncertainty-aware write gate

Volvence 应吸收 Bayesian update 的一个核心：不是所有 feedback 都同等更新。

owner update 至少应区分：

- evidence strength：证据强度。
- uncertainty type：epistemic / aleatoric / systematic。
- update confidence：是否足以写 durable memory。
- plasticity：允许快速改写还是需要保护旧状态。
- clarification need：证据不足时是否应提问，而不是更新。

这可以补强 R-PE 和 R5/R6 之间的连接。

## 2. 逐条证伪检查

### R1：多时间尺度学习

**结论：被加强，未被证伪。**

BOLT 线索强调每次 forward pass 后更新 latent memory，看似挑战 session/background 分层。但相邻文献显示 fast update 只是一个时间尺度：

- Continuous Latent Contexts 支持 fast latent state。
- Palimpsa 说明 fast fixed memory 有遗忘/干扰问题。
- Continual learning / gradient reconstruction 说明长期稳定需要 replay / regularization / consolidation。

所以结论是：BOLT-like update 可作为 `online-fast`，不能替代 `session-medium` 和 `background-slow`。

### R2：稳定基底 + 自适应控制器

**结论：强支持。**

PFN、Distribution Transformers、Clarke 优化线、JMHL subnetwork inference 都在说同一件事：把 expensive learning / inference 前移到离线训练阶段，在线只更新有界状态或局部子空间。没有一条核心证据支持“生产环境实时微调全量 LLM 权重”。

BOLT 若采用 latent memory + amortised encoder，正是 R2 的一个外部佐证。

### R3/R4：latent control space above token space

**结论：支持，但要求更精确区分 memory latent 与 control latent。**

Continuous Latent Contexts、Memory-Based Meta-Learning、Distribution Transformers 都支持 token 之上的 latent state。但它们的 latent state 不全是 control state：

- Distribution Transformers 的 latent 是 distribution component tokens。
- Continuous Latent Contexts 的 latent 是 online algorithm state。
- Volvence 的 `z_t / beta_t` 是 temporal controller state。

因此不能把 BOLT latent memory 直接等同于 ETA `z_t`。正确关系是：BOLT-like updater 可更新某个 owner 的 belief state；ETA controller 消费 owner snapshot 的 compact advisory，再选择 `z_t`。

### R-PE：prediction error 是原始学习信号

**结论：支持，但要求 typed evidence 更严格。**

PFN 和 memory-based meta-learning 以 log loss / sequential prediction error 为训练基础；Bayesian update 本身就是从 evidence 修正 belief。这与 R-PE 同向。

风险在于 BOLT 摘要说 “based on user feedback”。用户反馈不是天然 PE：

- 显式纠正可能是 task PE。
- 沉默/跳过可能是 ambiguous evidence。
- 关系反应可能是 self/relationship PE。
- 恶意反馈可能是 poisoned evidence。

所以 BOLT 方向要求 Volvence 更严格区分 raw feedback、typed evidence、prediction error、credit aggregation。

### R5/R6：记忆连续谱与慢反思

**结论：被加强，未被单一 latent memory 证伪。**

固定容量 latent memory 对 fast adaptation 很有价值，但 Palimpsa 和 continual learning 文献正说明：固定容量记忆会遇到灾难性遗忘和灾难性记住。长期关系、承诺、边界、common ground 不能只存在一个 hidden vector。

Volvence 的 CMS 分层仍成立：

- transient / online latent：快速 posterior cache。
- session episodic：可回放的事件与 evidence。
- durable semantic：慢层确认后的事实。
- derived indexes：检索和控制辅助。

BOLT-like memory 可加入第一层或 owner-private state，不能替代后三层。

### R7：任务/关系双轨

**结论：暂无直接证伪，反而暴露 BOLT 可能不足。**

BOLT 摘要强调 user workflow and task contexts，偏任务个性化。公开线索没有显示它区分 task feedback 与 relationship feedback。对 companion / digital organism，单轨 user feedback updater 有明显风险：把用户的情绪反应、边界信号、任务偏好混进同一 posterior。

Volvence 的 self/relationship track 仍必要。

### R8/R11：快照契约与内部状态可发布

**结论：Volvence 的系统约束未被论文反驳；反而是产品化 BOLT 必须补的层。**

学术论文可以把 latent memory 当黑箱；产品系统不能。若某个 hidden state 决定长期人格、关系、偏好和行动策略，它必须有 owner、summary、confidence、rollback evidence。

所以 BOLT-like updater 必须被包在 owner 内部，跨模块只发布 snapshot。否则会违反 R8，成为全局第二状态源。

### R15：可解释迁移与回滚

**结论：被加强。**

Clarke 的 update-rule 线和 Bayesian online update 都说明 update operator 会高度影响长期行为。任何引入 BOLT-like update 的路径必须：

- 先 SHADOW。
- 保存 old/new snapshot diff。
- 记录 evidence provenance。
- 有 exit condition。
- 有回滚策略。

这不是额外治理，而是 live-learning 的最低工程要求。

## 3. 是否证伪我方某些方案

### 3.1 “单一 latent memory state 能替代 CMS 分层”？

**否。**

证据只支持固定容量 latent state 可高效实现 online-fast；没有证据支持它能承载长期社会事实、关系连续性、承诺、边界、common ground。Palimpsa 和 continual learning 反而说明固定容量记忆需要 consolidation。

### 3.2 “每次 forward 都更新”是否证伪 session/background 分层？

**否。**

每次 forward 更新是 fast posterior update；它不处理历史审计、冲突消解、慢反思、durable promotion。Volvence 应吸收 fast update，而不是放弃慢层。

### 3.3 “Bayesian updater 足以替代 PE/credit/evaluation 链”？

**否。**

Bayesian updater 需要 evidence；R-PE 定义 evidence 如何从 prediction mismatch 产生。credit 是 PE 聚合，evaluation 是 readout/gate。BOLT-like updater 应消费这条链的 typed output，而不是替代它。

### 3.4 “Transformer learned inference 足以替代 owner contracts”？

**否。**

学术模型可以端到端；产品系统需要可解释、可回滚、可审计。Transformer updater 是 owner 内部实现，不是跨模块契约。

## 4. 推荐后续工程路线

### Phase A：Spec 补充

新增或扩展一个 spec 片段，暂名：

```text
docs/specs/owner-local-belief-update.md
```

内容：

- owner-local posterior state 的定义。
- typed evidence / PE 输入。
- uncertainty fields。
- snapshot publication。
- SHADOW 对照与 rollback gate。

### Phase B：最小 SHADOW 实验

选择一个低风险 owner，例如 `user_model` 的 stable preference confidence 或 `relationship_state` 的 rupture/repair tendency，做一个非生成式 updater：

```text
heuristic owner update
vs
Bayesian/ensemble posterior update shadow
```

只比较 snapshot diff、prediction calibration、false update rate，不改变线上行为。

### Phase C：接入 ETA

只有当 owner snapshot 可信后，ETA 才消费其 compact advisory。禁止 ETA 直接读 updater hidden state。

## 5. 总结

BOLT 方向不是 Volvence 的替代品，而是 Volvence 内部某些 owner 的候选 fast update kernel。它外部加强了我们对 R1/R2/R3/R4/R-PE/R5/R8/R15 的判断，同时暴露一个需要补强的形式化缺口：owner-local Bayesian belief update。

一句话：

> BOLT-like 技术值得吸收为“有界、owner-local、uncertainty-aware 的 online-fast posterior updater”；不应升级为全局记忆、全局控制器或跨模块隐式状态。
