# ETA Stage-3 P1 等价性诊断

> 范围：只归因，不重判已封存的 Stage-3 `kill-eta`，不改 production WiringLevel。

## 单页结论

- 主归因：`incentive-bypass-via-free-bias`
- exact-entry probe：0.391（chance 0.125，Gate-2 参考 0.944）
- bias-only 改善回收：0.963；zero-z 改善回收：0.614
- cyclic-permuted-z distortion penalty：-0.007
- oracle subgoal F1 − action-change F1：-0.068

## 匹配控制

| seed | mode | heldout D | zero-z D | permuted-z D | action F1 | oracle F1 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bias-only | 1.231 | — | — | 0.000 | 0.000 |
| 0 | full | 1.156 | 1.670 | 1.151 | 0.502 | 0.367 |
| 1 | bias-only | 1.231 | — | — | 0.000 | 0.000 |
| 1 | full | 1.190 | 1.791 | 1.190 | 0.000 | 0.000 |
| 2 | bias-only | 1.231 | — | — | 0.000 | 0.000 |
| 2 | full | 1.190 | 1.725 | 1.175 | 0.335 | 0.264 |

## 决策边界

P1 只能说明信息是否死在 exact entry、free bias 是否绕开 z、以及 z 是否具有因果作用。任何读数都不撤销 Stage-3 verdict；忠实 ETA rewrite 必须另立 claim / prereg。
