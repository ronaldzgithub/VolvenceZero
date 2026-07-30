# Gate 2 v36 live 根因：recent-k 控制历史诊断

## 1. 冻结事实

- v35 `promotion_allowed=true` 与 v36
  `shadow_observation_passed=false` 均为锁定历史结论，不覆盖、不重判。
- v36 失败发生在 `vz-runtime` evidence harness 的闭环组合协议，不等同于
  train-only selector 本身失效，也未进入真实 session/live wiring。
- v36 每一步把全部历史 decoded control 逐维求和，再把总和重新施加到当前
  更长 prefix。单步 selector 的训练支持域没有覆盖范数持续增长的组合控制。
- 已观察 artifact 显示 selector aggregate norm 随步数从约 `1.1` 增长到
  `3–5`；前两步总体仍为正，长历史阶段开始出现 fresh route 翻负。

## 2. 本收敛包的唯一变量

只验证 v36 预注册失败路径中的第一个方向：

- `k=1`：当前 forward 只激活最近一个 committed control；
- `k=2`：当前 forward 只激活最近两个 committed controls；
- v36 full-history 保持默认兼容行为，不在本包内修改 selector state features。

`committed_control_count` 继续记录轨迹累计提交数；新增
`active_control_count` 与 `committed_control_window`，明确每一步实际参与求和
的控制数。参数只允许 `1 / 2 / full-history`，非法值 fail loudly。

## 3. 开发诊断与选择规则

旧 v36 train / validation / confirmation 路线已经被观察，只能作为
development diagnostic：

1. 分别运行 k=1 与 k=2，同一 Qwen2.5-0.5B、CPU、full-width 896、
   max-prefix 8、seed 0 配置；
2. 每个 split 计算 selector-minus-zero、selector-minus-permutation 与
   selected step mean；
3. 若 k=1 在 train / validation / confirmation 三个 split 上三项均为正，
   选择 k=1；否则仅当 k=2 全部为正时选择 k=2；
4. 两档均失败，则 recent-k 方向止损，下一轮只能验证 v36 已预注册的
   committed-control summary state features；
5. 本轮不生成 formal promotion verdict，不把旧路线重放写成 fresh evidence。

## 4. 正式后续门

开发诊断选出唯一 k 后，才允许预注册新 schema 与 fresh validation /
locked confirmation routes。正式门仍要求 3 seeds 且三分区
selector-minus-zero、selector-minus-permutation、selected step mean 全正。
正式门通过前：

- `counterfactual_action_selector_live_injection=disabled`；
- runtime SHADOW/live wiring 保持冻结；
- v35 仅保留 open-loop causal-supported 结论。

## 5. 验证

- `pytest packages/vz-runtime/tests/test_eta_residual_causal_controls.py`
- `ruff check` 本包改动的 runtime、script 与 test 路径
- `git diff --check`
- 两次 single-seed development diagnostic（k=1、k=2）；不运行 3-seed
  formal suite，因为本轮复用了已观察的 v36 routes。

## 6. 回滚

删除 recent-k 参数、诊断 exporter 与脚本入口即可恢复原 v36 full-history
行为；默认调用路径从始至终保持 full-history，因此无需数据迁移。

## 7. 结果（2026-07-30）

两档均使用与 v36 相同的 Qwen2.5-0.5B、CPU、activation width 896、
max-prefix 8、train epochs 2、seed 0 和 28 条 route；selector fingerprint
均为 `ef360e0e…`，basis fingerprint 均为 `326aecdd…`，各自产出 483 条
三臂记录且全部 side-effect free。

| window | split | selector−zero | selector−permutation | step mean |
|---|---|---:|---:|---:|
| full-history (v36) | train | +0.126917 | +0.058562 | +0.022563 |
|  | validation | +0.104359 | **−0.040979** | +0.017393 |
|  | confirmation | +0.060983 | +0.054931 | +0.010164 |
| k=1 | train | +0.143859 | +0.127748 | +0.025575 |
|  | validation | **−0.010886** | +0.014666 | **−0.001814** |
|  | confirmation | +0.045450 | +0.049543 | +0.007575 |
| k=2 | train | +0.108693 | +0.079976 | +0.019323 |
|  | validation | +0.074078 | +0.091545 | +0.012346 |
|  | confirmation | +0.060443 | +0.034801 | +0.010074 |

结论：

- full-history 的 validation 失败确由控制历史组合失配主导；它允许任意
  permutation 历史通过无界累积获得比 selector 更高的偶然增益。
- k=1 把 validation aggregate norm 均值压到 `0.738`，修复了
  selector−permutation 符号，但发生欠注入，selector−zero 与 step mean
  翻负，因此不选择。
- k=2 的 validation aggregate norm 均值为 `1.368`，三项正式开发指标在
  train / validation / confirmation 全正，按冻结规则选为唯一 formal
  候选。
- heldout 在 k=2 仍为负（selector−zero `−0.058184`、
  selector−permutation `−0.057883`），这是已观察 development 风险，
  不能被三分区开发门掩盖；必须用新的 fresh routes 与 3 seeds 决定是否
  允许 runtime SHADOW。

本轮没有 formal promotion 或 SHADOW admission；live injection 继续
disabled。汇总 artifact：
`artifacts/eta_gate2_v36_recent_k_root_cause_development_20260730/`。
