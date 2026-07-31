# 机制改造战役总对账

Gate 9、1、6、4、10 的 owner 级实现缺口已逐一修复，并在既有冻结门槛下重跑。
五个包的 mechanism gates 全部通过，但 causal verdict 全部为
`not-supported`，因此五个 kill condition 均已触发。

- Gate 9：慢动量已真实进入 M3 更新，但 tracking/recovery 劣于 plain；
  production `slow_gain=0`。
- Gate 1：PE 已真实改变 ndim code，但 next-session loss 方向为负；
  production modulation flag 关闭。
- Gate 6：context prototype 已进入 CMS meta-init，且 benefit 改为 matched
  loss delta，但相对 copy-init 负迁移；production 使用 copy-init。
- Gate 4：固定 segment 加权已替换为 learned label-utility，但不省标签；
  learned selector 不晋升。
- Gate 10：training/eval 已共享 structural objective，遗忘与 rollback 受控，
  但 held-out transfer 仍为负；只保留 review-only 闭环。

Gate 7 不重跑，因为 Gate 1 修复后只有负向而非正向信号证据。#92 的
causal-supported 集合仍为 Gate 2/8/11，longitudinal-supported 集合仍为
Gate 8/11；共同等级保持 `mechanism-supported`，`thesis_retained=false`，
不授权 production/live promotion。
