# forecast_skill FAIL — 根因诊断（不改写判词）

判词工件：`report.json`（`forecast_skill=false`，stat=-0.0038，p=0.584；
`pe_discrimination=true`，signed_reward p≈1e-4；`cross_process_recovery=true`；
`external_outcome_channel=true`）。

## 根因探针（决定性）

对照实验：同一合成基底 Brain，三连 `run_test` 全失败 vs 全成功，读
`next_prediction.predicted_task_progress`：

- 全成功轨迹 bets: [0.4245, 0.3216, 0.3700, 0.3700]
- 全失败轨迹 bets: [0.4245, 0.3451, 0.3913, 0.3913]

失败流的预报**不低于**成功流 —— PE owner 的前向预测面在合成基底上不消费
`execution_result` / 环境结局证据。预报无技能是**预测头的属性**，不是环境
信号密度问题：丰富环境（如补 util 快速测试）不会改变本判词。

## 定性

- 结算面（ActualOutcome / signed_reward）分辨力极强 —— 外部结局通道、
  bias 表、结算链全部有刻度。
- 预报面（next_prediction）与证据流脱钩 —— 与七日 formal 的
  「仪器无刻度」同类，但在 Packet 1 成本（分钟级、零 formal 预算）被暴露。

## 修复路线（各自独立收敛包，不在本 lane 内 hack）

1. **Publisher 侧丰富**（正路）：在 `vz-cognition` PE owner 的预测头内
   把 `execution_result` / 近期 `EnvironmentOutcome` 证据纳入
   `predicted_task_progress` 的特征面 —— owner 内改动，一包一契约。
2. **真实基底重跑**：真实 substrate（文本特征反映事件流）+ 冻结 API 手
   的轨迹重跑本判词；本判词 scope 已限定为
   （synthetic substrate × scripted trajectories）。

## 对 lane 进度的含义

Packet 1 的机器（观察者、双下注、结算读出、置换检验、恢复探针）已验证；
`forecast_skill` 在当前配置下如实 FAIL。Packet 2+（记忆注入 / steelman 双门）
的机器照常构建；其 formal 判词本就依赖冻结 API 手，届时预报轴按上面路线
先过分辨力预检再进入 prereg。
