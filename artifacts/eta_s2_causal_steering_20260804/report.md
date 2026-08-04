# ETA S2 无 bias 因果残差 steering

> 范围：heldout cumulative-prefix 上的 matched causal evidence；不训练参数、不安装 artifact、不改 production WiringLevel。

## 结论

- S2 primary admission：`FAIL`
- primary scale：`0.50 × cap`
- failed conditions：`('plus-vs-noop-effect', 'plus-vs-minus-effect', 'plus-vs-shuffled-effect', 'route-win-rate', 'bootstrap-lower-positive')`

## 剂量与对照

| scale | +vs noop (95% CI) | +vs minus (95% CI) | +vs shuffled (95% CI) | route wins noop/minus/shuffle |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0022 [-0.0064, 0.0115] | 0.0114 [-0.0067, 0.0301] | 0.0039 [-0.0053, 0.0125] | 0.542 / 0.667 / 0.625 |
| 0.50 | -0.0007 [-0.0179, 0.0181] | 0.0283 [-0.0070, 0.0671] | 0.0071 [-0.0103, 0.0239] | 0.458 / 0.667 / 0.625 |
| 1.00 | -0.0243 [-0.0573, 0.0135] | 0.0914 [0.0175, 0.1721] | 0.0165 [-0.0162, 0.0480] | 0.208 / 0.750 / 0.667 |

## 守门边界

- 主判只读 0.50×cap；0.25/1.00 仅 dose diagnostic，不能救主判。
- target-plus 必须同时优于 noop、sign reversal、shuffled axes，且 route bootstrap 下界为正。
- 本结果不自动授权 S3 或 production promotion；后续仍需独立契约。
