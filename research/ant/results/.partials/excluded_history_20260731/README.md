# Ecology 历史 journal：EXCLUDED

本目录只用于保存 2026-07-31 终局战役 D3 / ecology E4 要求的历史原始证据，
**不属于任何正式运行的 progress root，不得续跑、恢复、晋级或用于改变预注册判词**。

- `v30_mps_reference/`：从 `/private/tmp/volvence-ecology-v30-worktree` 抢救的
  55 局历史 journal。日志首行明确记录 `device=mps:0`；MPS 设备口径未校准，
  因此只能用于历史审计，`decision_use=EXCLUDED`。
- `v31_polluted_ep23/`：旧 v31 journal，停在已提交 23 局，且双槽 checkpoint
  已越过并覆盖站 1 边界；它早于新同物理预注册与 dwell 收敛包，
  因此不能恢复为正式 station 运行，`decision_use=EXCLUDED`。

唯一可用于新判定的同物理运行必须使用独立 preregistration、独立空 progress
目录及其源码树 binding。逐文件字节数与 SHA-256 固定在 `manifest.json`；
归档后校验必须逐项相等。
