# ETA S1 冻结残差读出

> 范围：只验证 full-width residual 的线性可读性；不构成 causal steering 证据，不安装 artifact，不改 production WiringLevel。

## 结论

- S1 admission：`PASS`
- heldout accuracy：`0.9833` （chance `0.1250`，majority `0.2274`）
- early / late：`0.9896` / `0.9720`
- train / heldout gap：`0.0167`
- mean / min score margin：`0.8013` / `0.0012`
- artifact：`frozen-residual-readout.v1:086a8f3dc3e6270c8181231e711fe538a0e25c128bfd72d5054c83bc9f977df8`

## 固定几何

- layer：`(20,)`；width：`(896,)`
- classes：`8`；train / heldout rows：`551` / `299`
- axis：每类 effective weight 减其他类均值后 L2 normalize；不含 bias。

## 下一门

只有 S1 PASS 才允许用本 artifact 另行预注册 S2 的 +axis / −axis / noop / shuffled-axis matched controls。
