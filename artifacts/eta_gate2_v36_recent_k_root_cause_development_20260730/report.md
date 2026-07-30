# Gate 2 v36 recent-k 根因开发诊断

本包只复用已观察的 v36 routes 定位闭环控制历史组合根因，不是 fresh
evidence，不产生 formal promotion 或 SHADOW admission。

两档均使用 Qwen2.5-0.5B、CPU、activation width 896、max-prefix 8、
train epochs 2、seed 0。每档生成 483 条 selector / zero-control /
permutation-null 记录；selector fingerprint `ef360e0e…`、control basis
fingerprint `326aecdd…` 与锁定 v36 一致。

| window | split | selector−zero | selector−permutation | selected step mean |
|---|---|---:|---:|---:|
| k=1 | train | +0.143859 | +0.127748 | +0.025575 |
|  | validation | **−0.010886** | +0.014666 | **−0.001814** |
|  | confirmation | +0.045450 | +0.049543 | +0.007575 |
| k=2 | train | +0.108693 | +0.079976 | +0.019323 |
|  | validation | +0.074078 | +0.091545 | +0.012346 |
|  | confirmation | +0.060443 | +0.034801 | +0.010074 |

k=1 修复了 v36 validation 的 permutation-null 翻负，但形成欠注入；
k=2 九项开发门全部为正，按预注册规则成为下一 fresh formal 包的唯一候选。

已观察 development-heldout 在 k=2 仍为负：selector−zero
`−0.058184`、selector−permutation `−0.057883`、selected step mean
`−0.009697`。因此 k=2 尚不能进入 runtime SHADOW 或 live wiring。

下一步必须为 k=2 新建从未观察的 validation 与 locked confirmation routes，
并执行 seeds 0/1/2。正式门通过前，v35 open-loop causal verdict 保留，
`counterfactual_action_selector_live_injection=disabled`。

并发运行审计：较早 k=1 目录在 diagnostic-only exporter 修复前生成，混有
通用 paper-suite 附件，记为 `invalid-pre-fix-bundle`；修复后的干净 k=1
目录为权威输入。较早完成的干净 k=2 目录为权威输入，本轮稍后完成的 k=2
五个诊断文件与其逐字节一致，记为 `invalid-duplicate-not-admitted`。
