# Known Architecture Debt

> Status: tracked, not blocking
> Last updated: 2026-05-05

本文档记录已知但暂不处理的架构债。每条都经过评估：**不处理短期不会导致系统行为错误**，但**中长期会影响可演化性或可调试性**。新增条目时参照相同格式：路径 / 问题 / 风险 / 触发条件 / 推荐修法。

---

## 1. `interlocutor/__init__.py` duck-typed 多 owner 重建

- **路径**：`packages/vz-cognition/src/volvence_zero/interlocutor/__init__.py`
- **问题**：大量 `getattr` / duck typing 从 `regime / dual_track / evaluation / memory / commitment / PE` 快照里拼 `InterlocutorReadoutContext`。消费者在重建多个 owner 的状态摘要。
- **违反**：R8（"谁拥有数据谁描述"，消费者不重建生产者内部状态）
- **短期风险**：低。当前上游 snapshot 字段稳定时不会出错。
- **触发条件**：某个上游 owner 改字段名或字段语义 → interlocutor 会静默降级到默认值，社交 / 对话 readout 看起来还能跑，但上下文会读错，关系状态和表达策略逐渐漂。
- **推荐修法**：各 owner 发布自己的 compact interlocutor readout 字段；或设一个明确的 `InterlocutorContextModule` owner，不再在下游猜字段。
- **优先级**：**剩余债务中最高**。

## 2. `application/runtime.py` 硬编码 regime id 语义映射

- **路径**：`packages/vz-application/src/volvence_zero/application/runtime.py`
- **问题**：`_regime_bonus(regime_id, {"repair_and_deescalation": ..., "emotional_support": ..., "problem_solving": ...})` 等处把 regime id 映射成 risk / domain scoring bonus。application 层成了 regime 语义的第二解释者。
- **违反**：R8 + `no-keyword-matching-hacks`（用字符串字典驱动逻辑）
- **短期风险**：低-中。regime 集合稳定时不会爆。
- **触发条件**：regime 模板改名、新增或删除 regime → application 静默失配，行为漂移且不易发现。
- **推荐修法**：`RegimeSnapshot` 发布 application-facing hints（`risk_bonus / domain_bonus`），或 regime 包内提供单一 readout / mapping；application 只消费。
- **优先级**：中。

## 3. 三套语义 embedding stub 分叉

- **路径**：
  - `packages/vz-cognition/src/volvence_zero/dual_track/core.py`（`_semantic_embedding`，`% 37`）
  - `packages/vz-cognition/src/volvence_zero/evaluation/semantic_readouts.py`（`% 41`）
  - `packages/vz-application/src/volvence_zero/application/runtime.py`（`_semantic_embedding`，`dim=8`）
- **问题**：三套字符级 token + hash embedding，modulus / dim / prototype 字符串都不一致。同一句用户文本在不同模块得到**不可比**的语义压力分数。
- **违反**：SSOT（同一概念多处实现，不一致）
- **短期风险**：低。单模块内部一致，不崩。
- **触发条件**：跨模块联调 / 评估分数联立分析时 → 分数不可比，容易误判。
- **推荐修法**：抽一个 shared `semantic_embedding_stub`（或放进 `vz-contracts`），三处只引用同一函数；或升级为真正的 embedding 并单一缓存。
- **优先级**：中-低。

## 4. `EvaluationBackbone` 类型入口不干净

- **路径**：大量文件从 `volvence_zero.evaluation.backbone` 引入纯类型（`EvaluationSnapshot` / `EvaluationScore` 等），而 SSOT 已在 `evaluation/types.py`
- **问题**：consumer 只需要类型却绑定到含整棵 `EvaluationBackbone` 实现的模块，扩大加载图。
- **违反**：简洁性，非 R8 硬违反。
- **短期风险**：低。功能正常。
- **触发条件**：后续再拆分 evaluation 内部时易造成循环 import。
- **推荐修法**：机械收敛：把所有 `from volvence_zero.evaluation.backbone import <type>` 改为 `from volvence_zero.evaluation.types import <type>` 或 `from volvence_zero.evaluation import <type>`。
- **优先级**：低（纯维护性）。

## 5. `joint_loop` 与 runtime 主链共享 owner 实例

- **路径**：
  - 生产者：`packages/vz-runtime/src/volvence_zero/agent/session.py`（`AgentSessionRunner.__init__`）
  - 消费者：`packages/vz-temporal/src/volvence_zero/joint_loop/runtime.py`（`ETANLJointLoop.run_cycle`）
- **问题**：`_memory_store` / `_evaluation_backbone` / `_world_temporal_policy` / `_self_temporal_policy` / `_default_residual_runtime` 是同一实例被 runtime 主链和 `ETANLJointLoop` 同时持有并写入。属于"第二编排面"代码 pattern。
- **违反**：R8 精神（但在具体实现上已用 docstring 契约 + `TRAINING WRITEBACK PHASE` 注释块 + 契约测试 `test_joint_loop_shares_owner_instances_with_runtime_by_design` 固化边界）
- **短期风险**：低。turn 内是顺序执行而非并发，debug 现在有明确可视化的阶段边界。
- **触发条件**：有人把 writeback 逻辑加到 `TRAINING WRITEBACK PHASE` 注释块之外 → 重新变回"不可追踪"状态。契约测试只能测实例共享关系，测不出"phase block 之外的 mutation"。
- **推荐修法**：彻底方案是把 joint-loop post-propagate 的 owner writeback 搬到 runtime 编排层，joint-loop 只发 `JointCycleReport` typed proposals。这会破坏当前"在线 adaptation 立刻生效"的 pattern，需要重新设计 apply phase。不建议在产品迭代压力下做，等 NL 多时间尺度 apply phase 需要重新规划时一并做。
- **优先级**：低（已有契约测试兜底，边际收益低于成本）。

---

## 已关闭的债务（参考）

这些在 2026-05-04 至 2026-05-05 的 SSOT 收敛中已修完，留作对照：

- ~~`credit/gate.py -> temporal_types` 上游边界未声明~~
- ~~`SEMANTIC_OWNER_SLOTS` 在 `dual_track` 和 `semantic_state` 双源（dual_track 漏 `open_loop`）~~
- ~~`ReflectionEngine.apply(regime_module=...)` 直接持有并调用 `RegimeModule`~~
- ~~`memory/store.py` 解析 peer snapshot 内部字段拼 retrieval facets（temporal / dual_track / PE）~~
- ~~`EvaluationSnapshot.alerts` 文本子串驱动 regime / reflection / credit gate 控制逻辑~~
- ~~regime scoring 在 stable task opener 被 `guided_exploration` 抢占~~
- ~~super-loop diversity penalty 单峰不收敛 + `xfail` 的 `coding-regime.bs`~~

---

## 维护规则

1. 新加架构债时，先问自己：**"不改会死人吗？"**
   - 如果"短期风险"是"高"或"会爆"，不要写进这里，直接修。
   - 如果确实是"能跑 + 长期影响可演化性"，写进这里。
2. 每条都要有 **触发条件**。没有触发条件的债 = 不是债，是 preference。
3. 关闭条目时把它移到"已关闭的债务"段落，别直接删，留作 pattern 参考。