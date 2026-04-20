# Prediction Error 主链 Spec

> Status: draft
> Last updated: 2026-04-20
> 对应需求: R-PE

## 要解决的问题

如何把“预测 -> 实际结果 -> prediction error”从辅助日志提升为正式运行时主链，使它成为后续 credit / memory / temporal / regime / reflection 的共同学习原语？

## 关键不变量

- prediction error / LSS 是原始学习信号，不是可选诊断信息
- 系统必须显式发布 prediction chain，而不是只在下游隐式近似
- evaluation 是 prediction error 的 readout / gate 层，不是学习源头
- credit 是 prediction error 的聚合 / 审计层，不是学习源头
- prediction error 必须以 machine-readable 多维结构对外发布，而不是只剩一条文本描述

## 工程挑战

- 定义最小但稳定的 prediction chain 公共契约
- 保持 prediction error 的唯一 owner，避免各 consumer 自己重建 outcome mismatch
- 处理首轮 bootstrap 与跨轮 carryover，不制造同轮自因果闭环
- 让 downstream owner 能直接消费 task / relationship / regime / action 四维误差，而不需要重新解析文本

## 算法候选

来自 `docs/next_gen_emogpt.md`：

- **R-PE**：prediction error 是原始学习信号，evaluation/credit/reward 都是其下游读数或聚合
- **NL / LSS**：local surprise signal 是对预测与现实偏差的局部刻画
- **ETA**：时间抽象控制和 delayed outcome 学习应围绕 latent action 的后果误差展开，而不是只看 token 级局部损失

## 接口契约

**消费的输入**：

- `substrate` 快照：提供 turn-level semantic feature surface
- `evaluation` 快照：提供 family-level当前 readout，辅助构造 next-turn prediction
- `dual_track` 快照：提供 world/self tension 与 track-level state
- `regime` 快照：提供当前 regime 效果与稳定性线索

**产出的输出**：

- `prediction_error` 快照：`PredictionErrorSnapshot`
  - `evaluated_prediction`
  - `actual_outcome`
  - `next_prediction`
  - `error`
  - `turn_index`
  - `bootstrap`

**当前实现口径**：

- 正式 owner 为 `PredictionErrorModule`
- 公共 `error` 当前固定四个维度：
  - `task_error`
  - `relationship_error`
  - `regime_error`
  - `action_error`
- 聚合读数最小固定为：
  - `magnitude`
  - `signed_reward`
- `bootstrap=True` 表示当前 turn 尚无可结算的上一轮 prediction；下游不应把这类快照当作真实 learning evidence
- live runtime 中，部分 consumer 会把 `prediction_error` 当作“上一轮结算出的 carryover signal”，以维持单轮 DAG 和 owner 边界

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.9 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时 | 通过独立 slot 发布正式 prediction chain |
| 依赖 | 双轨学习 | task / relationship 维度误差需要双轨状态 |
| 依赖 | 认知 Regime | regime stability / action payoff 的一部分来自 regime owner 发布状态 |
| 被依赖 | 信用分配与自修改 | credit 是 prediction error 的聚合与审计层 |
| 被依赖 | 连续记忆系统 | memory owner 用 PE 调整写入、promotion threshold 和 retrieval facets |
| 被依赖 | 时间抽象与内部控制 | temporal owner 用 PE 调节 controller update 与 schedule 选择 |
| 被依赖 | 评估体系 | evaluation 把 PE 作为结构化 readout 和 benchmark 证据输入 |
| 被依赖 | 认知 Regime | regime owner 用 delayed / per-dimension PE 更新 historical effectiveness |
| 被依赖 | 慢反思路径 | reflection 将 PE 作为 tensions、lessons 和 policy consolidation 的正式输入 |

## 变更日志

- 2026-04-20: 初始版本。将 `prediction_error` 从 credit/evaluation 的上游设计原则提升为独立能力域 spec，固定主链契约 `evaluated_prediction -> actual_outcome -> next_prediction -> error`
